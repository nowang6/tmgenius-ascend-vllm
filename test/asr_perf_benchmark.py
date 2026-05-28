import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import ssl
import struct
import time
import wave
from typing import Dict, List, Optional


# ============================================================
# Minimal WebSocket client (stdlib only, no third-party deps)
# ============================================================

class _SimpleWS:
    """Minimal WebSocket client — asyncio + ssl, no third-party packages."""

    class ConnectionClosed(Exception):
        """Server closed the connection."""
        pass

    def __init__(self, reader, writer):
        self._reader = reader
        self._writer = writer

    @classmethod
    async def connect(cls, url, open_timeout=5.0):
        if url.startswith("ws://"):
            rest = url[5:]
            use_ssl = False
        elif url.startswith("wss://"):
            rest = url[6:]
            use_ssl = True
        else:
            raise ValueError(f"Unsupported WS URL scheme: {url}")

        if "/" in rest:
            host_port, path = rest.split("/", 1)
            path = "/" + path
        else:
            host_port = rest
            path = "/"

        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 443 if use_ssl else 80

        ssl_ctx = ssl.create_default_context() if use_ssl else None
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ssl_ctx),
            timeout=open_timeout,
        )

        # WebSocket upgrade handshake
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )
        writer.write(req.encode())
        await writer.drain()

        resp = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"), timeout=open_timeout
        )
        resp_text = resp.decode(errors="replace")
        if "101" not in resp_text:
            raise Exception(
                f"WebSocket handshake failed: {resp_text.split(chr(13)+chr(10))[0]}"
            )

        # Verify accept key
        expected = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode()
        for line in resp_text.split("\r\n"):
            if line.lower().startswith("sec-websocket-accept:"):
                got = line.split(":", 1)[1].strip()
                if got != expected:
                    raise Exception("WebSocket accept key mismatch")
                break

        return cls(reader, writer)

    async def send(self, data: str):
        payload = data.encode()
        frame = self._frame(0x1, payload, mask=True)
        self._writer.write(frame)
        await self._writer.drain()

    async def recv(self) -> str:
        while True:
            header = await self._reader.readexactly(2)
            opcode = header[0] & 0xF
            plen = header[1] & 0x7F

            if plen == 126:
                plen = struct.unpack("!H", await self._reader.readexactly(2))[0]
            elif plen == 127:
                plen = struct.unpack("!Q", await self._reader.readexactly(8))[0]

            masked = (header[1] & 0x80) != 0
            mask = await self._reader.readexactly(4) if masked else b""

            payload = await self._reader.readexactly(plen)
            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x1:  # text
                return payload.decode()
            elif opcode == 0x8:  # close
                code = struct.unpack("!H", payload[:2])[0] if len(payload) >= 2 else 1000
                raise _SimpleWS.ConnectionClosed(code)
            elif opcode == 0x9:  # ping → pong (client MUST mask)
                pong = self._frame(0xA, payload, mask=True)
                self._writer.write(pong)
                await self._writer.drain()
            # pong / continuation — continue

    async def close(self):
        try:
            self._writer.write(self._frame(0x8, b"", mask=True))
            await self._writer.drain()
        except Exception:
            pass
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass

    @staticmethod
    def _frame(opcode, payload, mask):
        frame = bytes([0x80 | opcode])
        plen = len(payload)
        if plen < 126:
            frame += bytes([(0x80 if mask else 0x00) | plen])
        elif plen < 65536:
            frame += bytes([(0x80 if mask else 0x00) | 126]) + struct.pack("!H", plen)
        else:
            frame += bytes([(0x80 if mask else 0x00) | 127]) + struct.pack("!Q", plen)

        if mask:
            mk = os.urandom(4)
            frame += mk
            payload = bytes(b ^ mk[i % 4] for i, b in enumerate(payload))

        return frame + payload

class SegmentMetrics:
    def __init__(self, bg_ms, ed_ms, text, recv_audio_clock_ms):
        self.bg_ms = bg_ms
        self.ed_ms = ed_ms
        self.text = text
        # recv_audio_clock_ms: result received at what "audio clock" position (ms).
        # i.e. (recv_monotonic - first_chunk_send_monotonic) * 1000
        # This allows direct comparison with ed_ms to detect falling behind.
        self.recv_audio_clock_ms = recv_audio_clock_ms
        self.asr_ms = 0.0
        self.total_ms = 0.0
        self.segment_e2e_ms = 0.0

class ConnectionMetrics:
    def __init__(self, conn_id):
        self.conn_id = conn_id
        self.segments = []
        self.bottleneck_count = 0
        self.success = False
        self.error = None
        self.bottleneck_details = []

class ResourceSample:
    def __init__(self, timestamp, cpu_percent, mem_percent, mem_used_mb):
        self.timestamp = timestamp
        self.cpu_percent = cpu_percent
        self.mem_percent = mem_percent
        self.mem_used_mb = mem_used_mb

class LevelResult:
    def __init__(self, level):
        self.level = level
        self.connections = []
        self.resource_samples = []
        self.wall_time = 0.0

def load_wav_pcm(path, chunk_samples=640):
    with wave.open(path, 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("WAV must be 16kHz, 16-bit, mono")
        frames = wf.readframes(wf.getnframes())
    
    samples = struct.unpack(f'<{len(frames)//2}h', frames)
    chunks = []
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i:i+chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = chunk + (0,) * (chunk_samples - len(chunk))
        chunks.append(struct.pack(f'<{len(chunk)}h', *chunk))
    return chunks

class SystemMonitor:
    def __init__(self):
        self.last_cpu_times = self._get_cpu_times()
        self.last_time = time.time()

    def _get_cpu_times(self):
        try:
            with open('/proc/stat', 'r') as f:
                line = f.readline()
                parts = line.split()
                if parts[0] == 'cpu':
                    return sum(float(x) for x in parts[1:8]), float(parts[4]) # total, idle
        except:
            pass
        return 0.0, 0.0

    def _get_mem_info(self):
        mem_total = 0
        mem_available = 0
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1])
                    elif line.startswith('MemAvailable:'):
                        mem_available = int(line.split()[1])
        except:
            pass
        if mem_total == 0:
            return 0.0, 0.0
        
        used = mem_total - mem_available
        mem_percent = (used / mem_total) * 100.0
        mem_used_mb = used / 1024.0
        return mem_percent, mem_used_mb

    def sample(self):
        now = time.time()
        cpu_times = self._get_cpu_times()
        
        cpu_percent = 0.0
        if self.last_cpu_times[0] > 0 and cpu_times[0] > self.last_cpu_times[0]:
            total_diff = cpu_times[0] - self.last_cpu_times[0]
            idle_diff = cpu_times[1] - self.last_cpu_times[1]
            if total_diff > 0:
                cpu_percent = 100.0 * (total_diff - idle_diff) / total_diff

        self.last_cpu_times = cpu_times
        self.last_time = now

        mem_percent, mem_used_mb = self._get_mem_info()
        return ResourceSample(now, cpu_percent, mem_percent, mem_used_mb)

async def _resource_monitor(level_result, stop_event, interval=1.0):
    monitor = SystemMonitor()
    # Initial sample to set baseline for CPU
    monitor.sample()
    await asyncio.sleep(0.1)
    
    while not stop_event.is_set():
        sample = monitor.sample()
        level_result.resource_samples.append(sample)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

async def _run_connection(conn_id, url, chunks, chunk_samples, args, start_delay):
    metrics = ConnectionMetrics(conn_id)
    if start_delay > 0:
        await asyncio.sleep(start_delay)

    trace_id = f"bench_{conn_id}"
    biz_id = f"bench_{conn_id}"

    try:
        ws = await _SimpleWS.connect(url, open_timeout=args.open_timeout)
        try:

            # --- Handshake (status=0) ---
            handshake = {
                "header": {
                    "traceId": trace_id,
                    "bizId": biz_id,
                    "status": 0,
                }
            }
            await ws.send(json.dumps(handshake))

            # Wait for handshake response (status=0)
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.recv_timeout)
                resp = json.loads(raw)
                if resp.get("header", {}).get("status") != 0:
                    raise Exception(f"Handshake rejected: {resp}")
            except asyncio.TimeoutError:
                raise Exception("Handshake timeout")

            # ---- Per-chunk send timestamps (client clock as authoritative time base) ----
            chunk_duration = chunk_samples / 16000.0
            chunk_ms = chunk_duration * 1000.0
            send_times = []  # type: List[float]

            # VAD post-padding frames (ASR_PAD_FRAMES * HOP_SIZE samples)
            PAD_SAMPLES = 5 * 640
            PAD_MS = PAD_SAMPLES * 1000.0 / 16000.0  # 200ms

            async def send_audio():
                t_base = time.monotonic()
                for i, chunk in enumerate(chunks):
                    target_time = t_base + i * chunk_duration
                    now = time.monotonic()
                    delay = target_time - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    else:
                        # Even when behind schedule, yield to let recv_results() run.
                        # Without this, drain() may not truly yield and recv starves.
                        await asyncio.sleep(0)

                    b64 = base64.b64encode(chunk).decode()
                    req = {
                        "header": {
                            "traceId": trace_id,
                            "bizId": biz_id,
                            "status": 1,
                        },
                        "payload": {
                            "audio": {
                                "audio": b64,
                                "encoding": None,
                            }
                        },
                    }
                    await ws.send(json.dumps(req))
                    send_times.append(time.monotonic())

                # Send EOS (status=2)
                req = {
                    "header": {
                        "traceId": trace_id,
                        "bizId": biz_id,
                        "status": 2,
                    }
                }
                await ws.send(json.dumps(req))

            async def recv_results():
                while True:
                    try:
                        msg = await ws.recv()
                    except _SimpleWS.ConnectionClosed:
                        break

                    # Capture recv timestamp immediately after socket read
                    recv_ts = time.monotonic()
                    resp = json.loads(msg)
                    header = resp.get("header", {})

                    if header.get("status") == 2:
                        _extract_segment(resp, send_times, chunk_ms, PAD_MS, metrics, recv_ts)
                        break

                    _extract_segment(resp, send_times, chunk_ms, PAD_MS, metrics, recv_ts)

            await asyncio.gather(send_audio(), recv_results())
            metrics.success = True

        finally:
            await ws.close()

    except Exception as e:
        metrics.error = str(e)

    return metrics


def _extract_segment(resp, send_times, chunk_ms, pad_ms, metrics, recv_time=None):
    """Extract a segment result and compute client-clock-based E2E delay.

    E2E = receive_time - send_time_of_last_speech_frame
    where send_time_of_last_speech_frame is looked up via the client-side
    send timestamp records, mapped from the server-reported ed_ms (minus
    VAD post-padding frames).
    """
    result = resp.get("payload", {}).get("result", {})
    ws_list = result.get("ws", [])
    if not ws_list:
        return
    cw_list = ws_list[0].get("cw", [])
    if not cw_list:
        return

    if recv_time is None:
        recv_time = time.monotonic()
    bg_ms = result.get("bg", 0)
    ed_ms = result.get("ed", 0)
    text = cw_list[0].get("w", "")

    # Parse server-side timing from header.message (JSON: {"asr_ms": ..., "total_ms": ...})
    server_asr_ms = 0.0
    server_total_ms = 0.0
    header = resp.get("header", {})
    msg_str = header.get("message", "")
    if msg_str and msg_str.startswith("{"):
        try:
            timing = json.loads(msg_str)
            server_asr_ms = float(timing.get("asr_ms", 0))
            server_total_ms = float(timing.get("total_ms", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Actual speech end = ed_ms minus VAD post-padding frames
    speech_end_ms = max(0.0, ed_ms - pad_ms)

    # Find the send timestamp of the chunk containing this audio position
    chunk_idx = int(speech_end_ms / chunk_ms)
    if send_times and chunk_idx < len(send_times):
        send_t = send_times[chunk_idx]
        e2e_ms = (recv_time - send_t) * 1000.0
    else:
        send_t = send_times[0] if send_times else recv_time
        e2e_ms = (recv_time - send_t) * 1000.0 - speech_end_ms

    # Convert recv_time to audio-clock ms (relative to first chunk send time)
    audio_clock_base = send_times[0] if send_times else recv_time
    recv_audio_clock_ms = (recv_time - audio_clock_base) * 1000.0

    seg = SegmentMetrics(bg_ms, ed_ms, text, recv_audio_clock_ms)
    seg.asr_ms = server_asr_ms
    seg.total_ms = server_total_ms
    seg.segment_e2e_ms = e2e_ms
    metrics.segments.append(seg)

def _detect_bottlenecks(metrics):
    """Detect bottlenecks: segment N result arrived after segment N+1 audio was fully sent.

    Both recv_audio_clock_ms and ed_ms are in the same coordinate system:
    milliseconds relative to the start of audio sending. If seg_N's result
    arrives at audio-clock time T, and T > seg_N+1.ed_ms, it means the system
    is falling behind — by the time we got seg_N's result, all audio for
    seg_N+1 had already been transmitted.
    """
    metrics.segments.sort(key=lambda x: x.bg_ms)
    for i in range(len(metrics.segments) - 1):
        seg_n = metrics.segments[i]
        seg_n1 = metrics.segments[i + 1]

        recv_clock_ms = seg_n.recv_audio_clock_ms
        next_seg_ed_ms = seg_n1.ed_ms

        if recv_clock_ms > next_seg_ed_ms:
            lag_ms = recv_clock_ms - next_seg_ed_ms
            metrics.bottleneck_count += 1
            metrics.bottleneck_details.append({
                "seg_idx": i,
                "recv_clock_ms": recv_clock_ms,
                "next_ed_ms": next_seg_ed_ms,
                "lag_ms": lag_ms,
            })

async def run_level(level, url, chunks, chunk_samples, args):
    print(f"\n--- Running Level: {level} Connections ---")
    result = LevelResult(level)
    stop_event = asyncio.Event()
    
    monitor_task = asyncio.create_task(_resource_monitor(result, stop_event))
    
    tasks = []
    level_start = time.monotonic()
    
    for i in range(level):
        delay = i * (args.stagger_ms / 1000.0)
        task = asyncio.create_task(_run_connection(i, url, chunks, chunk_samples, args, delay))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    result.wall_time = time.monotonic() - level_start
    
    stop_event.set()
    await monitor_task
    
    for m in results:
        _detect_bottlenecks(m)
        result.connections.append(m)
        
    return result

def percentile(data, p):
    if not data:
        return 0.0
    s_data = sorted(data)
    idx = int(math.ceil(p / 100.0 * len(s_data))) - 1
    idx = max(0, min(idx, len(s_data) - 1))
    return s_data[idx]

def _display_width(s: str) -> int:
    """Calculate display width accounting for CJK double-width characters."""
    width = 0
    for ch in s:
        if '\u4e00' <= ch <= '\u9fff' or '\u3000' <= ch <= '\u303f' or '\uff00' <= ch <= '\uffef':
            width += 2
        else:
            width += 1
    return width


def _pad(s: str, width: int, align: str = "<") -> str:
    """Pad string to target display width, respecting CJK double-width."""
    cur = _display_width(s)
    pad_n = max(0, width - cur)
    if align == ">":
        return " " * pad_n + s
    elif align == "^":
        left = pad_n // 2
        return " " * left + s + " " * (pad_n - left)
    return s + " " * pad_n


def _table(headers: List[str], rows: List[List[str]], col_aligns: Optional[List[str]] = None) -> List[str]:
    """Build a formatted ASCII table with box-drawing borders.

    col_aligns: list of '<' (left), '>' (right), '^' (center) per column.
    """
    n_cols = len(headers)
    if col_aligns is None:
        col_aligns = ["<"] * n_cols

    col_widths = [_display_width(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], _display_width(cell))

    for i in range(n_cols):
        col_widths[i] += 2

    def _sep(left, mid, right, fill="─"):
        parts = [fill * w for w in col_widths]
        return left + mid.join(parts) + right

    def _row(cells, aligns):
        parts = []
        for i, cell in enumerate(cells):
            inner = _pad(cell, col_widths[i] - 2, aligns[i])
            parts.append(" " + inner + " ")
        return "│" + "│".join(parts) + "│"

    lines = []
    lines.append(_sep("┌", "┬", "┐"))
    # Header row — always center-aligned
    lines.append(_row(headers, ["^"] * n_cols))
    lines.append(_sep("├", "┼", "┤"))
    for row in rows:
        lines.append(_row(row, col_aligns))
    lines.append(_sep("└", "┴", "┘"))
    return lines


def generate_report(results, args, audio_duration_ms):
    W = 78
    lines = []

    # Title
    lines.append("╔" + "═" * W + "╗")
    title = "ASR 端到端性能基准测试报告"
    lines.append("║" + _pad(title, W, "^") + "║")
    lines.append("╚" + "═" * W + "╝")
    lines.append("")

    # Meta info
    lines.append(f"  生成时间 : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  服务地址 : {args.url}")
    lines.append(f"  音频文件 : {os.path.basename(args.audio)} ({audio_duration_ms/1000.0:.2f}s)")
    lines.append(f"  并发级别 : {args.levels}")
    lines.append("")

    # Section 1: Summary
    lines.append("─" * W)
    lines.append("  [一] 汇总总览")
    lines.append("─" * W)
    lines.append("")

    headers = ["并发", "成功", "失败", "耗时(s)", "CPU%avg", "CPU%max", "MEM avg", "MEM max",
               "E2E avg", "E2E P99", "瓶颈", "RTF"]
    col_aligns = [">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">", ">"]
    rows = []

    for res in results:
        success = sum(1 for c in res.connections if c.success)
        fail = len(res.connections) - success

        cpu_avg = sum(s.cpu_percent for s in res.resource_samples) / max(1, len(res.resource_samples))
        cpu_max = max((s.cpu_percent for s in res.resource_samples), default=0.0)

        mem_avg = sum(s.mem_used_mb for s in res.resource_samples) / max(1, len(res.resource_samples))
        mem_max = max((s.mem_used_mb for s in res.resource_samples), default=0.0)

        all_e2e = []
        bottlenecks = 0
        for c in res.connections:
            bottlenecks += c.bottleneck_count
            for s in c.segments:
                all_e2e.append(s.segment_e2e_ms)

        e2e_avg = sum(all_e2e) / max(1, len(all_e2e))
        e2e_p99 = percentile(all_e2e, 99)
        rtf = res.wall_time / (audio_duration_ms / 1000.0)

        rows.append([
            str(res.level),
            str(success),
            str(fail),
            f"{res.wall_time:.2f}",
            f"{cpu_avg:.1f}%",
            f"{cpu_max:.1f}%",
            f"{mem_avg:.0f}M",
            f"{mem_max:.0f}M",
            f"{e2e_avg:.0f}ms",
            f"{e2e_p99:.0f}ms",
            str(bottlenecks),
            f"{rtf:.2f}",
        ])

    lines.extend(_table(headers, rows, col_aligns))
    lines.append("")

    # Section 2: Per-segment E2E vs total_ms breakdown per concurrency level
    lines.append("─" * W)
    lines.append("  [二] 分段延迟对比 (每段语音 E2E vs 服务端耗时)")
    lines.append("─" * W)
    lines.append("")
    lines.append("  E2E     = 语音结束帧发出 → 收到结果 (含VAD断句等待+ASR+ITN+排序)")
    lines.append("  total_ms = 服务端 ASR+ITN 处理耗时 (不含VAD等待)")
    lines.append("  差值    ≈ VAD 静默断句等待 + 结果排序等待")
    lines.append("")

    has_any_segments = False
    for res in results:
        # Collect per-segment-index data across all connections in this level
        seg_e2e = {}    # type: Dict[int, List[float]]
        seg_total = {}  # type: Dict[int, List[float]]
        seg_asr = {}    # type: Dict[int, List[float]]
        seg_bg = {}     # type: Dict[int, List[int]]
        seg_ed = {}     # type: Dict[int, List[int]]
        max_seg_idx = -1
        for c in res.connections:
            for idx, s in enumerate(c.segments):
                seg_e2e.setdefault(idx, []).append(s.segment_e2e_ms)
                seg_bg.setdefault(idx, []).append(s.bg_ms)
                seg_ed.setdefault(idx, []).append(s.ed_ms)
                if s.total_ms > 0:
                    seg_total.setdefault(idx, []).append(s.total_ms)
                if s.asr_ms > 0:
                    seg_asr.setdefault(idx, []).append(s.asr_ms)
                max_seg_idx = max(max_seg_idx, idx)

        if max_seg_idx < 0:
            continue
        has_any_segments = True

        lines.append(f"  ▸ 并发级别 {res.level} ({sum(1 for c in res.connections if c.success)}/{len(res.connections)} 成功)")
        lines.append("")

        seg_headers = ["段#", "区间(ms)", "E2E avg(ms)", "total avg(ms)", "差值(ms)", "ASR avg(ms)"]
        seg_aligns = [">", "<", ">", ">", ">", ">"]
        seg_rows = []
        for seg_idx in range(max_seg_idx + 1):
            e2e_vals = seg_e2e.get(seg_idx, [])
            total_vals = seg_total.get(seg_idx, [])
            asr_vals = seg_asr.get(seg_idx, [])
            bg_vals = seg_bg.get(seg_idx, [])
            ed_vals = seg_ed.get(seg_idx, [])

            e2e_avg = sum(e2e_vals) / len(e2e_vals) if e2e_vals else 0
            total_avg = sum(total_vals) / len(total_vals) if total_vals else 0
            asr_avg = sum(asr_vals) / len(asr_vals) if asr_vals else 0
            diff = e2e_avg - total_avg if total_vals else 0

            # Use the most common bg/ed (or first connection's value)
            bg_rep = int(sum(bg_vals) / len(bg_vals)) if bg_vals else 0
            ed_rep = int(sum(ed_vals) / len(ed_vals)) if ed_vals else 0

            seg_rows.append([
                f"第{seg_idx}段",
                f"{bg_rep}-{ed_rep}",
                f"{e2e_avg:.0f}" if e2e_vals else "-",
                f"{total_avg:.0f}" if total_vals else "-",
                f"{diff:.0f}" if total_vals else "-",
                f"{asr_avg:.0f}" if asr_vals else "-",
            ])

        # Summary row
        all_e2e = [v for vals in seg_e2e.values() for v in vals]
        all_total = [v for vals in seg_total.values() for v in vals]
        all_asr = [v for vals in seg_asr.values() for v in vals]
        total_e2e_avg = sum(all_e2e) / len(all_e2e) if all_e2e else 0
        total_total_avg = sum(all_total) / len(all_total) if all_total else 0
        total_asr_avg = sum(all_asr) / len(all_asr) if all_asr else 0
        total_diff = total_e2e_avg - total_total_avg if all_total else 0
        seg_rows.append([
            "整体avg",
            "-",
            f"{total_e2e_avg:.0f}" if all_e2e else "-",
            f"{total_total_avg:.0f}" if all_total else "-",
            f"{total_diff:.0f}" if all_total else "-",
            f"{total_asr_avg:.0f}" if all_asr else "-",
        ])

        lines.extend(_table(seg_headers, seg_rows, seg_aligns))
        lines.append("")

    if not has_any_segments:
        lines.append("  无有效分段数据")
        lines.append("")

    # Section 3: Detailed per-level report
    lines.append("─" * W)
    lines.append("  [三] 各并发级别详细报告")
    lines.append("─" * W)

    for res in results:
        lines.append("")
        lines.append(f"  ┌── 并发级别 {res.level} ──┐")
        lines.append("")

        # 4.1 Resource usage
        lines.append("  4.1 资源使用 (采样间隔 ~1s)")
        if res.resource_samples:
            r_headers = ["秒", "CPU %", "MEM MB"]
            r_aligns = [">", ">", ">"]
            r_rows = []
            for i, s in enumerate(res.resource_samples):
                r_rows.append([str(i), f"{s.cpu_percent:.1f}", f"{s.mem_used_mb:.0f}"])
            lines.extend(["  " + l for l in _table(r_headers, r_rows, r_aligns)])
        else:
            lines.append("      无采样数据")
        lines.append("")

        # 4.2 E2E latency stats
        lines.append("  4.2 E2E 延迟分布")
        all_e2e = []
        for c in res.connections:
            for s in c.segments:
                all_e2e.append(s.segment_e2e_ms)

        if all_e2e:
            lines.append(f"      avg : {sum(all_e2e)/len(all_e2e):>8.1f} ms")
            lines.append(f"      P50 : {percentile(all_e2e, 50):>8.1f} ms")
            lines.append(f"      P90 : {percentile(all_e2e, 90):>8.1f} ms")
            lines.append(f"      P99 : {percentile(all_e2e, 99):>8.1f} ms")
            lines.append(f"      max : {max(all_e2e):>8.1f} ms")
        else:
            lines.append("      无有效数据")
        lines.append("")

        # 4.3 Bottleneck detection
        lines.append("  4.3 瓶颈检测")
        total_bn = sum(c.bottleneck_count for c in res.connections)
        if total_bn == 0:
            lines.append("      未检测到瓶颈")
        else:
            lines.append(f"      总计瓶颈次数: {total_bn}")
            for c in res.connections:
                if c.bottleneck_count > 0:
                    lines.append(f"      连接 {c.conn_id}: {c.bottleneck_count} 次")
                    for d in c.bottleneck_details[:3]:
                        lines.append(
                            f"        └ 段{d['seg_idx']}结果到达时(audio clock {d['recv_clock_ms']:.0f}ms)"
                            f" 已超过下一段结束位置({d['next_ed_ms']:.0f}ms)，滞后 {d['lag_ms']:.0f}ms"
                        )
                    if len(c.bottleneck_details) > 3:
                        lines.append(f"        └ ... 还有 {len(c.bottleneck_details)-3} 条")
        lines.append("")

        # 4.4 Segment details
        lines.append("  4.4 分段明细")
        shown = False
        for c in res.connections:
            if c.error:
                lines.append(f"      连接 {c.conn_id}: ✗ {c.error}")
                shown = True
            elif c.segments:
                lines.append(f"      连接 {c.conn_id}: {len(c.segments)} 段")
                seg_headers = ["#", "区间(ms)", "E2E", "svr_ms", "文本"]
                seg_aligns = [">", "<", ">", ">", "<"]
                seg_rows = []
                display_segs = c.segments[:8] if c.bottleneck_count > 0 else c.segments[:5]
                for i, s in enumerate(display_segs):
                    seg_rows.append([
                        str(i),
                        f"{s.bg_ms}-{s.ed_ms}",
                        f"{s.segment_e2e_ms:.0f}",
                        f"{s.total_ms:.0f}" if s.total_ms > 0 else "-",
                        s.text[:20] + ("..." if len(s.text) > 20 else ""),
                    ])
                if len(c.segments) > len(display_segs):
                    seg_rows.append(["...", "...", "...", "...", f"(共{len(c.segments)}段)"])
                lines.extend(["      " + l for l in _table(seg_headers, seg_rows, seg_aligns)])
                lines.append("")
                shown = True
        if not shown:
            lines.append("      无分段数据")
        lines.append("")

    # Footer
    lines.append("─" * W)
    lines.append("  报告结束")
    lines.append("─" * W)

    report_text = "\n".join(lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nReport saved to {args.output}")


async def amain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8856/tuling/ast/v3")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--levels", default="1,8,16,32,64,128,256,512")
    parser.add_argument("--chunk-samples", type=int, default=640)
    parser.add_argument("--open-timeout", type=float, default=5.0)
    parser.add_argument("--recv-timeout", type=float, default=10.0)
    parser.add_argument("--stagger-ms", type=float, default=2.0)
    parser.add_argument("--cooldown", type=float, default=10.0)
    parser.add_argument("--output", default="asr_perf_report.txt")
    
    args = parser.parse_args()
    
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    
    print(f"Loading audio {args.audio}...")
    chunks = load_wav_pcm(args.audio, args.chunk_samples)
    audio_duration_ms = len(chunks) * args.chunk_samples / 16.0
    print(f"Audio loaded. Duration: {audio_duration_ms/1000.0:.2f}s, Chunks: {len(chunks)}")
    
    results = []
    
    for i, level in enumerate(levels):
        res = await run_level(level, args.url, chunks, args.chunk_samples, args)
        results.append(res)
        
        if i < len(levels) - 1:
            print(f"Cooldown for {args.cooldown} seconds...")
            await asyncio.sleep(args.cooldown)
            
    generate_report(results, args, audio_duration_ms)

def main():
    asyncio.run(amain())

if __name__ == "__main__":
    main()
