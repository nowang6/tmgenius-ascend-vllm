#!/usr/bin/env python3
"""
WebSocket ASR 并发压力测试客户端。

功能与 Java 客户端一致：
  1. 建立 WebSocket 连接
  2. 发送握手帧 (status=0)
  3. 按 40ms 节奏流式发送音频帧 (status=1)
  4. 发送结束帧 (status=2)
  5. 接收服务端返回的识别结果
  6. 收到 status=2 后主动关闭连接

额外能力：
  - 详细的统计日志（每连接、全局汇总）
  - 异常分类计数
  - 服务端 ASR 耗时解析与统计
  - 输出格式方便粘贴给 AI 分析 bug

VAD 分段策略（动态停顿阈值）：
  - 0~20s 语音：停顿阈值从 VAD_PAUSE_MAX 线性递减至 VAD_PAUSE_MIN
  - 20~30s 语音：固定 VAD_PAUSE_MIN 停顿阈值
  - >30s 语音：到达 VAD_MAX_SPEECH 强制触发分段
  - <0.5s 短音频：抑制不转发
  分片数量取决于音频内容，不再是固定值。

用法：
    python test/ws_stress_test.py --url ws://localhost:8856/tuling/ast/v3 \
                                  --concurrency 50 \
                                  --audio 120报警电话16k.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import traceback
import wave
from dataclasses import dataclass, field
from typing import Optional

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.config import settings

# ============================================================
# 数据结构
# ============================================================

@dataclass
class SegmentResult:
    """单个识别结果片段。"""
    seg_id: int = 0
    bg_ms: int = 0
    ed_ms: int = 0
    text: str = ""
    status: int = 0          # header.status
    recv_time: float = 0.0   # monotonic timestamp
    asr_ms: float = 0.0      # 服务端 ASR 模型推理耗时(ms)
    total_ms: float = 0.0    # 服务端总处理耗时(ms) ASR+ITN

    @property
    def audio_duration_ms(self) -> float:
        """音频段时长（毫秒）。"""
        return max(0, self.ed_ms - self.bg_ms)


@dataclass
class ConnectionResult:
    """单个连接的完整结果。"""
    conn_id: int = 0
    trace_id: str = ""
    sid: str = ""

    # 时间线
    t_start: float = 0.0
    t_connected: float = 0.0
    t_handshake_sent: float = 0.0
    t_handshake_ack: float = 0.0
    t_first_audio_sent: float = 0.0
    t_last_audio_sent: float = 0.0
    t_end_sent: float = 0.0
    t_final_result: float = 0.0
    t_closed: float = 0.0

    # 发送统计
    chunks_sent: int = 0
    bytes_sent: int = 0

    # 接收到的结果
    segments: list[SegmentResult] = field(default_factory=list)

    # 错误
    error: str = ""
    error_type: str = ""  # timeout / connection_closed / refused / exception

    # 服务端错误消息
    server_errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return not self.error

    @property
    def has_final(self) -> bool:
        return any(s.status == 2 for s in self.segments)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    @property
    def zero_duration_segments(self) -> list[SegmentResult]:
        return [s for s in self.segments if s.bg_ms == 0 and s.ed_ms == 0 and s.status != 0]

    @property
    def e2e_latency(self) -> float:
        """端到端延迟：从开始到最后结果（秒）。"""
        if self.t_final_result > 0 and self.t_start > 0:
            return self.t_final_result - self.t_start
        return 0.0

    @property
    def handshake_latency(self) -> float:
        if self.t_handshake_ack > 0 and self.t_handshake_sent > 0:
            return self.t_handshake_ack - self.t_handshake_sent
        return 0.0


# ============================================================
# 音频加载
# ============================================================

def load_wav_pcm(path: str) -> bytes:
    """加载 WAV 文件，返回原始 PCM bytes (int16)。"""
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()}"
        assert wf.getsampwidth() == 2, f"Expected 16-bit"
        assert wf.getframerate() == 16000, f"Expected 16kHz"
        return wf.readframes(wf.getnframes())


# ============================================================
# 单连接测试逻辑
# ============================================================

async def run_single_connection(
    conn_id: int,
    url: str,
    pcm_bytes: bytes,
    chunk_samples: int,
    send_interval: float,
    hotwords: str,
    open_timeout: float,
    recv_timeout: float,
) -> ConnectionResult:
    """运行单个 WebSocket 连接的完整生命周期。"""
    result = ConnectionResult(conn_id=conn_id)
    result.trace_id = f"stress_test_{conn_id}_{int(time.time()*1000)}"
    result.t_start = time.monotonic()

    chunk_bytes = chunk_samples * 2  # int16 = 2 bytes per sample

    try:
        # ---- 连接 ----
        ws = await asyncio.wait_for(
            websockets.connect(
                url,
                ping_interval=settings.WS_PING_INTERVAL,
                ping_timeout=settings.WS_PING_TIMEOUT,
                close_timeout=10,
                max_size=10 * 1024 * 1024,
            ),
            timeout=open_timeout,
        )
        result.t_connected = time.monotonic()

        try:
            # ---- 握手帧 ----
            handshake_msg = {
                "header": {
                    "traceId": result.trace_id,
                    "appId": "stress_test",
                    "bizId": f"test_user_{conn_id}",
                    "status": 0,
                },
                "payload": {
                    "audio": {"audio": ""},
                    "text": {"text": hotwords} if hotwords else None,
                },
            }
            result.t_handshake_sent = time.monotonic()
            await ws.send(json.dumps(handshake_msg, ensure_ascii=False))

            # 等待握手响应
            raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
            result.t_handshake_ack = time.monotonic()
            resp = json.loads(raw)
            result.sid = resp.get("header", {}).get("sid", "")

            if resp.get("header", {}).get("code", -1) != 0:
                result.error = f"Handshake failed: {resp}"
                result.error_type = "handshake_error"
                await ws.close()
                return result

            # ---- 启动接收协程 ----
            recv_done = asyncio.Event()

            async def receiver():
                try:
                    while True:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                        t_recv = time.monotonic()
                        msg = json.loads(raw_msg)
                        header = msg.get("header", {})
                        payload = msg.get("payload", {})
                        res = payload.get("result", {})

                        code = header.get("code", 0)
                        if code != 0:
                            result.server_errors.append(
                                f"code={code}, msg={header.get('message', '')}"
                            )
                            if header.get("status") == 2:
                                recv_done.set()
                                return
                            continue

                        # 解析服务端耗时信息
                        asr_ms = 0.0
                        total_ms = 0.0
                        msg_field = header.get("message", "")
                        if msg_field and msg_field.startswith("{"):
                            try:
                                timing = json.loads(msg_field)
                                asr_ms = timing.get("asr_ms", 0.0)
                                total_ms = timing.get("total_ms", 0.0)
                            except (json.JSONDecodeError, TypeError):
                                pass

                        seg = SegmentResult(
                            seg_id=res.get("segId", -1),
                            bg_ms=res.get("bg", 0),
                            ed_ms=res.get("ed", 0),
                            text=_extract_text(res),
                            status=header.get("status", -1),
                            recv_time=t_recv,
                            asr_ms=asr_ms,
                            total_ms=total_ms,
                        )
                        result.segments.append(seg)

                        if header.get("status") == 2:
                            result.t_final_result = t_recv
                            recv_done.set()
                            return
                except asyncio.TimeoutError:
                    result.error = "Recv timeout waiting for server response"
                    result.error_type = "recv_timeout"
                    recv_done.set()
                except ConnectionClosed as e:
                    if not recv_done.is_set():
                        result.error = f"Connection closed during recv: code={e.code}, reason={e.reason}"
                        result.error_type = "connection_closed"
                    recv_done.set()

            recv_task = asyncio.create_task(receiver())

            # ---- 发送音频帧 ----
            offset = 0
            first_sent = False
            while offset < len(pcm_bytes):
                chunk = pcm_bytes[offset : offset + chunk_bytes]
                offset += chunk_bytes

                b64 = base64.b64encode(chunk).decode("ascii")
                audio_msg = {
                    "header": {
                        "traceId": result.trace_id,
                        "appId": "stress_test",
                        "bizId": f"test_user_{conn_id}",
                        "status": 1,
                    },
                    "payload": {"audio": {"audio": b64}},
                }
                await ws.send(json.dumps(audio_msg))
                result.chunks_sent += 1
                result.bytes_sent += len(chunk)

                if not first_sent:
                    result.t_first_audio_sent = time.monotonic()
                    first_sent = True

                # 模拟 40ms 实时节奏
                await asyncio.sleep(send_interval)

            result.t_last_audio_sent = time.monotonic()

            # ---- 结束帧 ----
            end_msg = {
                "header": {
                    "traceId": result.trace_id,
                    "appId": "stress_test",
                    "bizId": f"test_user_{conn_id}",
                    "status": 2,
                },
                "payload": {"audio": {"audio": ""}},
            }
            await ws.send(json.dumps(end_msg))
            result.t_end_sent = time.monotonic()

            # ---- 等待所有结果 ----
            try:
                await asyncio.wait_for(recv_done.wait(), timeout=recv_timeout)
            except asyncio.TimeoutError:
                if not result.error:
                    result.error = "Timeout waiting for final result after end frame"
                    result.error_type = "final_timeout"

            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass

        finally:
            try:
                await ws.close()
            except Exception:
                pass
            result.t_closed = time.monotonic()

    except asyncio.TimeoutError:
        result.error = "Connection open timeout"
        result.error_type = "connect_timeout"
        result.t_closed = time.monotonic()
    except ConnectionRefusedError:
        result.error = "Connection refused"
        result.error_type = "refused"
        result.t_closed = time.monotonic()
    except OSError as e:
        result.error = f"OS error: {e}"
        result.error_type = "os_error"
        result.t_closed = time.monotonic()
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.error_type = "exception"
        result.t_closed = time.monotonic()

    return result


def _extract_text(result_payload: dict) -> str:
    """从 result payload 提取识别文本。"""
    ws_list = result_payload.get("ws", [])
    texts = []
    for ws_item in ws_list:
        for cw in ws_item.get("cw", []):
            w = cw.get("w", "")
            if w:
                texts.append(w)
    return "".join(texts)


# ============================================================
# 并发调度
# ============================================================

async def run_stress_test(
    url: str,
    pcm_bytes: bytes,
    concurrency: int,
    chunk_samples: int,
    send_interval: float,
    hotwords: str,
    open_timeout: float,
    recv_timeout: float,
    stagger_ms: float,
) -> list[ConnectionResult]:
    """并发运行多个连接。"""

    tasks = []
    for i in range(concurrency):
        tasks.append(
            run_single_connection(
                conn_id=i,
                url=url,
                pcm_bytes=pcm_bytes,
                chunk_samples=chunk_samples,
                send_interval=send_interval,
                hotwords=hotwords,
                open_timeout=open_timeout,
                recv_timeout=recv_timeout,
            )
        )
        if stagger_ms > 0 and i < concurrency - 1:
            await asyncio.sleep(stagger_ms / 1000.0)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            cr = ConnectionResult(conn_id=i)
            cr.error = f"Task exception: {type(r).__name__}: {r}"
            cr.error_type = "task_exception"
            final.append(cr)
        else:
            final.append(r)
    return final


# ============================================================
# 统计工具
# ============================================================

def _percentile(sorted_list: list[float], p: float) -> float:
    """计算百分位数（输入须已排序）。"""
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * p / 100.0)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


# ============================================================
# 报告生成
# ============================================================

def generate_report(
    results: list[ConnectionResult],
    audio_duration: float,
    concurrency: int,
    url: str,
) -> str:
    """生成详细统计报告（文本格式，方便粘贴）。"""
    lines: list[str] = []
    L = lines.append

    L("=" * 80)
    L("ASR WebSocket 并发压力测试报告")
    L("=" * 80)
    L(f"服务地址:       {url}")
    L(f"并发数:         {concurrency}")
    L(f"音频时长:       {audio_duration:.2f}s")
    L(f"VAD 分段策略:   动态停顿阈值 ({settings.VAD_PAUSE_MAX}s→{settings.VAD_PAUSE_MIN}s 线性递减, {settings.VAD_MAX_SPEECH}s 强制触发)")
    L("")

    # ---- 总体统计 ----
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    L(f"--- 总体统计 ---")
    L(f"成功连接:   {len(success)} / {len(results)}")
    L(f"失败连接:   {len(failed)} / {len(results)}")
    L("")

    # ---- 错误分类 ----
    if failed:
        L(f"--- 错误分类 ---")
        error_types: dict[str, list[ConnectionResult]] = {}
        for r in failed:
            error_types.setdefault(r.error_type or "unknown", []).append(r)
        for etype, conns in sorted(error_types.items()):
            L(f"  {etype}: {len(conns)} 个")
            for c in conns[:3]:  # 每类最多展示3个样例
                L(f"    conn#{c.conn_id}: {c.error}")
            if len(conns) > 3:
                L(f"    ... 还有 {len(conns)-3} 个")
        L("")

    # ---- 分片统计 ----
    if success:
        L(f"--- 分片统计 (成功连接) ---")
        seg_counts = [r.total_segments for r in success]
        seg_dist: dict[int, int] = {}
        for c in seg_counts:
            seg_dist[c] = seg_dist.get(c, 0) + 1

        L(f"  分片数分布:")
        for count in sorted(seg_dist.keys()):
            bar = "█" * seg_dist[count]
            L(f"    {count:2d} 个分片: {seg_dist[count]:3d} 个连接  {bar}")

        # 0-0 空分片统计
        zero_dur_conns = [r for r in success if r.zero_duration_segments]
        if zero_dur_conns:
            L(f"")
            L(f"  ⚠ 出现 bg=0,ed=0 空分片的连接: {len(zero_dur_conns)} 个")
            for r in zero_dur_conns[:5]:
                for seg in r.zero_duration_segments:
                    L(f"    conn#{r.conn_id} seg#{seg.seg_id}: bg={seg.bg_ms} ed={seg.ed_ms} status={seg.status}")
            if len(zero_dur_conns) > 5:
                L(f"    ... 还有 {len(zero_dur_conns)-5} 个")
        L("")

    # ---- 是否收到 final (status=2) ----
    if success:
        has_final = [r for r in success if r.has_final]
        no_final = [r for r in success if not r.has_final]
        L(f"--- 终态 (status=2) ---")
        L(f"  收到终态:   {len(has_final)} / {len(success)}")
        L(f"  未收到终态: {len(no_final)} / {len(success)}")
        if no_final:
            for r in no_final[:5]:
                L(f"    conn#{r.conn_id}: 最后收到 status={r.segments[-1].status if r.segments else 'N/A'}")
        L("")

    # ---- 延迟统计 ----
    if success:
        L(f"--- 延迟统计 (成功连接) ---")
        hs_lats = [r.handshake_latency for r in success if r.handshake_latency > 0]
        e2e_lats = [r.e2e_latency for r in success if r.e2e_latency > 0]

        if hs_lats:
            hs_lats.sort()
            L(f"  握手延迟:  min={hs_lats[0]*1000:.0f}ms  "
              f"med={hs_lats[len(hs_lats)//2]*1000:.0f}ms  "
              f"max={hs_lats[-1]*1000:.0f}ms")
        if e2e_lats:
            e2e_lats.sort()
            L(f"  端到端:    min={e2e_lats[0]:.1f}s  "
              f"med={e2e_lats[len(e2e_lats)//2]:.1f}s  "
              f"max={e2e_lats[-1]:.1f}s")
        L("")

    # ---- ASR 服务端耗时统计 ----
    if success:
        all_asr_ms = []
        all_total_ms = []
        for r in success:
            for seg in r.segments:
                if seg.asr_ms > 0:
                    all_asr_ms.append(seg.asr_ms)
                if seg.total_ms > 0:
                    all_total_ms.append(seg.total_ms)

        if all_asr_ms:
            L(f"--- 服务端 ASR 模型耗时统计 ({len(all_asr_ms)} 个分段) ---")
            all_asr_ms.sort()
            avg_asr = sum(all_asr_ms) / len(all_asr_ms)
            L(f"  min={all_asr_ms[0]:.1f}ms  avg={avg_asr:.1f}ms  "
              f"med={_percentile(all_asr_ms, 50):.1f}ms  "
              f"P95={_percentile(all_asr_ms, 95):.1f}ms  "
              f"P99={_percentile(all_asr_ms, 99):.1f}ms  "
              f"max={all_asr_ms[-1]:.1f}ms")
            L("")

        if all_total_ms:
            L(f"--- 服务端总处理耗时统计 ({len(all_total_ms)} 个分段, ASR+ITN) ---")
            all_total_ms.sort()
            avg_total = sum(all_total_ms) / len(all_total_ms)
            L(f"  min={all_total_ms[0]:.1f}ms  avg={avg_total:.1f}ms  "
              f"med={_percentile(all_total_ms, 50):.1f}ms  "
              f"P95={_percentile(all_total_ms, 95):.1f}ms  "
              f"P99={_percentile(all_total_ms, 99):.1f}ms  "
              f"max={all_total_ms[-1]:.1f}ms")
            L("")

    # ---- 服务端错误消息 ----
    srv_err_conns = [r for r in results if r.server_errors]
    if srv_err_conns:
        L(f"--- 服务端错误消息 ---")
        for r in srv_err_conns[:10]:
            for e in r.server_errors:
                L(f"  conn#{r.conn_id}: {e}")
        if len(srv_err_conns) > 10:
            L(f"  ... 还有 {len(srv_err_conns)-10} 个连接有服务端错误")
        L("")

    # ---- 每连接明细（前 10 个 + 全部失败的） ----
    L(f"--- 连接明细 (前10个成功 + 全部失败) ---")
    shown = set()
    for r in (success[:10] + failed):
        if r.conn_id in shown:
            continue
        shown.add(r.conn_id)
        status_mark = "✓" if r.success else "✗"
        L(f"  [{status_mark}] conn#{r.conn_id:03d} sid={r.sid or 'N/A'}")
        L(f"      segments={r.total_segments} has_final={r.has_final} "
          f"chunks_sent={r.chunks_sent}")
        if r.handshake_latency > 0:
            L(f"      handshake={r.handshake_latency*1000:.0f}ms "
              f"e2e={r.e2e_latency:.1f}s")
        if r.error:
            L(f"      ERROR [{r.error_type}]: {r.error}")
        if r.segments:
            for seg in r.segments:
                dur = f"bg={seg.bg_ms}ms ed={seg.ed_ms}ms"
                timing = ""
                if seg.asr_ms > 0:
                    timing = f" asr={seg.asr_ms:.0f}ms total={seg.total_ms:.0f}ms"
                empty = " ⚠EMPTY" if seg.bg_ms == 0 and seg.ed_ms == 0 and seg.status != 0 else ""
                L(f"      seg#{seg.seg_id}: {dur} status={seg.status} "
                  f"text={seg.text!r}{timing}{empty}")
    L("")
    L("=" * 80)
    L("报告结束")
    L("=" * 80)

    return "\n".join(lines)


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ASR WebSocket 并发压力测试")
    parser.add_argument(
        "--url",
        default="ws://localhost:8856/tuling/ast/v3",
        help="WebSocket 服务地址",
    )
    parser.add_argument(
        "--audio",
        default=os.path.join(
            os.path.dirname(__file__), "..", "120报警电话16k.wav"
        ),
        help="测试音频文件路径 (PCM 16k 16bit mono WAV)",
    )
    parser.add_argument("--concurrency", "-c", type=int, default=50, help="并发连接数")
    parser.add_argument(
        "--chunk-samples", type=int, default=640,
        help="每次发送的样本数 (默认 640 = 40ms @ 16kHz)",
    )
    parser.add_argument(
        "--send-interval", type=float, default=0.04,
        help="发送间隔（秒），模拟实时 (默认 0.04 = 40ms)",
    )
    parser.add_argument("--hotwords", default="", help="热词 (逗号分隔)")
    parser.add_argument("--open-timeout", type=float, default=30.0, help="连接超时(秒)")
    parser.add_argument("--recv-timeout", type=float, default=120.0, help="接收超时(秒)")
    parser.add_argument(
        "--stagger-ms", type=float, default=50,
        help="连接间错开毫秒数 (默认 50ms)",
    )
    parser.add_argument("--output", "-o", default="", help="报告输出文件 (默认 stdout)")

    args = parser.parse_args()

    # 加载音频
    audio_path = os.path.abspath(args.audio)
    print(f"Loading audio: {audio_path}")
    pcm_bytes = load_wav_pcm(audio_path)
    audio_duration = len(pcm_bytes) / 2 / 16000  # int16 = 2 bytes/sample
    print(f"Audio duration: {audio_duration:.2f}s, PCM size: {len(pcm_bytes)} bytes")
    print(f"Starting {args.concurrency} concurrent connections to {args.url}")
    print(f"Stagger: {args.stagger_ms}ms between connections")
    print()

    # 运行测试
    t0 = time.monotonic()
    results = asyncio.run(
        run_stress_test(
            url=args.url,
            pcm_bytes=pcm_bytes,
            concurrency=args.concurrency,
            chunk_samples=args.chunk_samples,
            send_interval=args.send_interval,
            hotwords=args.hotwords,
            open_timeout=args.open_timeout,
            recv_timeout=args.recv_timeout,
            stagger_ms=args.stagger_ms,
        )
    )
    elapsed = time.monotonic() - t0
    print(f"Test completed in {elapsed:.1f}s\n")

    # 生成报告
    report = generate_report(results, audio_duration, args.concurrency, args.url)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
