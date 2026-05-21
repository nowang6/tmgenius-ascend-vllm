#!/usr/bin/env python3
"""
ASR 多并发时延分析测试。

在不同并发级别 (1, 10, 20, 30, 40, 50, 100, 200, 500) 下：
  - 发送同一音频至 ASR 服务
  - 从服务端响应中提取每个 VAD 分段的 ASR 模型处理耗时
  - 统计并输出详细报告（txt 格式）

用法：
    python test/asr_latency_profile.py \
        --url ws://localhost:8856/tuling/ast/v3 \
        --audio 120报警电话16k.wav \
        --output asr_latency_report.txt

    # 自定义并发级别
    python test/asr_latency_profile.py --levels 1,10,50,100

注意：
    - 服务端需部署带有耗时信息的版本（header.message 包含 asr_ms/total_ms）
    - 高并发测试需确保 MAX_CONNECTIONS 配置足够大
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.config import settings

# ============================================================
# 默认并发级别
# ============================================================

DEFAULT_LEVELS = [1, 10, 20, 30, 40, 50, 100, 200, 500]

# ============================================================
# 数据结构
# ============================================================

@dataclass
class SegmentTiming:
    """单个 VAD 分段的计时信息。"""
    conn_id: int = 0
    seg_id: int = 0
    bg_ms: int = 0
    ed_ms: int = 0
    text: str = ""
    status: int = 0
    # 服务端耗时（从 header.message 解析）
    asr_ms: float = 0.0
    total_ms: float = 0.0
    # 客户端侧时间戳
    recv_time: float = 0.0

    @property
    def audio_duration_ms(self) -> float:
        return max(0, self.ed_ms - self.bg_ms)

    @property
    def asr_rtf(self) -> float:
        """ASR 实时率：asr_ms / 音频时长。越小越好。"""
        if self.audio_duration_ms > 0 and self.asr_ms > 0:
            return self.asr_ms / self.audio_duration_ms
        return 0.0


@dataclass
class ConnectionTiming:
    """单个连接的计时汇总。"""
    conn_id: int = 0
    sid: str = ""
    success: bool = True
    error: str = ""

    t_start: float = 0.0
    t_connected: float = 0.0
    t_end_sent: float = 0.0
    t_final_result: float = 0.0
    t_closed: float = 0.0

    segments: list[SegmentTiming] = field(default_factory=list)

    @property
    def e2e_seconds(self) -> float:
        if self.t_final_result > 0 and self.t_start > 0:
            return self.t_final_result - self.t_start
        return 0.0


@dataclass
class LevelResult:
    """一个并发级别的汇总结果。"""
    concurrency: int = 0
    wall_time: float = 0.0
    connections: list[ConnectionTiming] = field(default_factory=list)


# ============================================================
# 音频加载
# ============================================================

def load_wav_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()}"
        assert wf.getsampwidth() == 2, f"Expected 16-bit"
        assert wf.getframerate() == 16000, f"Expected 16kHz"
        return wf.readframes(wf.getnframes())


# ============================================================
# 单连接
# ============================================================

async def _run_connection(
    conn_id: int,
    url: str,
    pcm_bytes: bytes,
    chunk_bytes: int,
    send_interval: float,
    open_timeout: float,
    recv_timeout: float,
) -> ConnectionTiming:
    ct = ConnectionTiming(conn_id=conn_id)
    ct.t_start = time.monotonic()
    trace_id = f"latency_profile_{conn_id}_{int(time.time()*1000)}"

    try:
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
        ct.t_connected = time.monotonic()

        try:
            # 握手
            handshake = {
                "header": {
                    "traceId": trace_id,
                    "appId": "latency_profile",
                    "bizId": f"profile_{conn_id}",
                    "status": 0,
                },
                "payload": {"audio": {"audio": ""}},
            }
            await ws.send(json.dumps(handshake))
            raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
            resp = json.loads(raw)
            ct.sid = resp.get("header", {}).get("sid", "")

            if resp.get("header", {}).get("code", -1) != 0:
                ct.success = False
                ct.error = f"Handshake failed: {resp}"
                return ct

            # 接收协程
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
                            if header.get("status") == 2:
                                recv_done.set()
                                return
                            continue

                        # 解析耗时
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

                        is_final = header.get("status") == 2
                        text = _extract_text(res)

                        # 纯信号帧（status=2 且无文本）不创建分段记录
                        if not is_final or text:
                            seg = SegmentTiming(
                                conn_id=conn_id,
                                seg_id=res.get("segId", -1),
                                bg_ms=res.get("bg", 0),
                                ed_ms=res.get("ed", 0),
                                text=text,
                                status=header.get("status", -1),
                                asr_ms=asr_ms,
                                total_ms=total_ms,
                                recv_time=t_recv,
                            )
                            ct.segments.append(seg)

                        if is_final:
                            ct.t_final_result = t_recv
                            recv_done.set()
                            return
                except asyncio.TimeoutError:
                    ct.error = "recv_timeout"
                    ct.success = False
                    recv_done.set()
                except ConnectionClosed:
                    if not recv_done.is_set():
                        ct.error = "connection_closed"
                        ct.success = False
                    recv_done.set()

            recv_task = asyncio.create_task(receiver())

            # 发送音频
            offset = 0
            while offset < len(pcm_bytes):
                chunk = pcm_bytes[offset : offset + chunk_bytes]
                offset += chunk_bytes
                b64 = base64.b64encode(chunk).decode("ascii")
                audio_msg = {
                    "header": {
                        "traceId": trace_id,
                        "appId": "latency_profile",
                        "bizId": f"profile_{conn_id}",
                        "status": 1,
                    },
                    "payload": {"audio": {"audio": b64}},
                }
                await ws.send(json.dumps(audio_msg))
                await asyncio.sleep(send_interval)

            # 结束帧
            end_msg = {
                "header": {
                    "traceId": trace_id,
                    "appId": "latency_profile",
                    "bizId": f"profile_{conn_id}",
                    "status": 2,
                },
                "payload": {"audio": {"audio": ""}},
            }
            await ws.send(json.dumps(end_msg))
            ct.t_end_sent = time.monotonic()

            try:
                await asyncio.wait_for(recv_done.wait(), timeout=recv_timeout)
            except asyncio.TimeoutError:
                if not ct.error:
                    ct.error = "final_timeout"
                    ct.success = False

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
            ct.t_closed = time.monotonic()

    except asyncio.TimeoutError:
        ct.error = "connect_timeout"
        ct.success = False
        ct.t_closed = time.monotonic()
    except ConnectionRefusedError:
        ct.error = "connection_refused"
        ct.success = False
        ct.t_closed = time.monotonic()
    except Exception as e:
        ct.error = f"{type(e).__name__}: {e}"
        ct.success = False
        ct.t_closed = time.monotonic()

    return ct


def _extract_text(result_payload: dict) -> str:
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

async def run_level(
    url: str,
    pcm_bytes: bytes,
    concurrency: int,
    chunk_bytes: int,
    send_interval: float,
    open_timeout: float,
    recv_timeout: float,
    stagger_ms: float,
) -> LevelResult:
    """运行一个并发级别的测试。"""
    t0 = time.monotonic()

    tasks = []
    for i in range(concurrency):
        tasks.append(
            _run_connection(
                conn_id=i,
                url=url,
                pcm_bytes=pcm_bytes,
                chunk_bytes=chunk_bytes,
                send_interval=send_interval,
                open_timeout=open_timeout,
                recv_timeout=recv_timeout,
            )
        )
        if stagger_ms > 0 and i < concurrency - 1:
            await asyncio.sleep(stagger_ms / 1000.0)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    connections = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            ct = ConnectionTiming(conn_id=i, success=False, error=str(r))
            connections.append(ct)
        else:
            connections.append(r)

    wall_time = time.monotonic() - t0
    return LevelResult(
        concurrency=concurrency,
        wall_time=wall_time,
        connections=connections,
    )


# ============================================================
# 统计工具
# ============================================================

def _percentile(sorted_list: list[float], p: float) -> float:
    if not sorted_list:
        return 0.0
    idx = int(len(sorted_list) * p / 100.0)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


def _stat_line(label: str, values: list[float]) -> str:
    """生成统计行。"""
    if not values:
        return f"  {label}: (无数据)"
    values_sorted = sorted(values)
    avg = sum(values_sorted) / len(values_sorted)
    return (
        f"  {label}: "
        f"min={values_sorted[0]:.1f}ms  "
        f"avg={avg:.1f}ms  "
        f"med={_percentile(values_sorted, 50):.1f}ms  "
        f"P95={_percentile(values_sorted, 95):.1f}ms  "
        f"P99={_percentile(values_sorted, 99):.1f}ms  "
        f"max={values_sorted[-1]:.1f}ms"
    )


# ============================================================
# 报告生成
# ============================================================

def generate_full_report(
    level_results: list[LevelResult],
    audio_duration: float,
    url: str,
    audio_path: str,
) -> str:
    lines: list[str] = []
    L = lines.append

    L("=" * 100)
    L("ASR 多并发时延分析报告")
    L("=" * 100)
    L(f"生成时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L(f"服务地址:     {url}")
    L(f"测试音频:     {audio_path}")
    L(f"音频时长:     {audio_duration:.2f}s")
    L(f"并发级别:     {', '.join(str(lr.concurrency) for lr in level_results)}")
    L(f"VAD 分段策略: 动态停顿阈值 ({settings.VAD_PAUSE_MAX}s→{settings.VAD_PAUSE_MIN}s 线性递减, {settings.VAD_MAX_SPEECH}s 强制触发)")
    L("")

    # ============================================================
    # 汇总表格
    # ============================================================
    L("=" * 100)
    L("一、汇总统计表")
    L("=" * 100)
    L("")
    L(f"{'并发':>6} | {'成功':>4} | {'失败':>4} | {'总耗时':>8} | "
      f"{'ASR avg':>10} | {'ASR med':>10} | {'ASR P95':>10} | {'ASR P99':>10} | {'ASR max':>10} | "
      f"{'Total avg':>10} | {'Total P95':>10}")
    L("-" * 120)

    for lr in level_results:
        ok = [c for c in lr.connections if c.success]
        fail = [c for c in lr.connections if not c.success]

        all_asr = []
        all_total = []
        for c in ok:
            for s in c.segments:
                if s.asr_ms > 0:
                    all_asr.append(s.asr_ms)
                if s.total_ms > 0:
                    all_total.append(s.total_ms)

        asr_sorted = sorted(all_asr) if all_asr else []
        total_sorted = sorted(all_total) if all_total else []

        asr_avg = f"{sum(asr_sorted)/len(asr_sorted):.1f}" if asr_sorted else "N/A"
        asr_med = f"{_percentile(asr_sorted, 50):.1f}" if asr_sorted else "N/A"
        asr_p95 = f"{_percentile(asr_sorted, 95):.1f}" if asr_sorted else "N/A"
        asr_p99 = f"{_percentile(asr_sorted, 99):.1f}" if asr_sorted else "N/A"
        asr_max = f"{asr_sorted[-1]:.1f}" if asr_sorted else "N/A"

        total_avg = f"{sum(total_sorted)/len(total_sorted):.1f}" if total_sorted else "N/A"
        total_p95 = f"{_percentile(total_sorted, 95):.1f}" if total_sorted else "N/A"

        L(f"{lr.concurrency:>6} | {len(ok):>4} | {len(fail):>4} | {lr.wall_time:>7.1f}s | "
          f"{asr_avg:>10} | {asr_med:>10} | {asr_p95:>10} | {asr_p99:>10} | {asr_max:>10} | "
          f"{total_avg:>10} | {total_p95:>10}")

    L("")

    # ============================================================
    # 每个并发级别的详细报告
    # ============================================================
    for lr in level_results:
        L("=" * 100)
        L(f"二、并发 {lr.concurrency} 详细报告")
        L("=" * 100)

        ok = [c for c in lr.connections if c.success]
        fail = [c for c in lr.connections if not c.success]

        L(f"  成功: {len(ok)}  失败: {len(fail)}  总耗时: {lr.wall_time:.1f}s")
        L("")

        # 错误汇总
        if fail:
            L(f"  --- 失败连接 ---")
            for c in fail[:10]:
                L(f"    conn#{c.conn_id}: {c.error}")
            if len(fail) > 10:
                L(f"    ... 还有 {len(fail)-10} 个")
            L("")

        # ASR 耗时统计
        all_asr = []
        all_total = []
        for c in ok:
            for s in c.segments:
                if s.asr_ms > 0:
                    all_asr.append(s.asr_ms)
                if s.total_ms > 0:
                    all_total.append(s.total_ms)

        if all_asr:
            L(f"  --- ASR 模型耗时 ({len(all_asr)} 个分段) ---")
            L(_stat_line("ASR模型", all_asr))
        if all_total:
            L(f"  --- 总处理耗时 ({len(all_total)} 个分段, ASR+ITN) ---")
            L(_stat_line("总处理", all_total))
        L("")

        # 分段数分布
        seg_counts = [len(c.segments) for c in ok]
        if seg_counts:
            seg_dist: dict[int, int] = {}
            for sc in seg_counts:
                seg_dist[sc] = seg_dist.get(sc, 0) + 1
            L(f"  --- 分段数分布 ---")
            for count in sorted(seg_dist.keys()):
                bar = "█" * min(seg_dist[count], 50)
                L(f"    {count:2d} 个分段: {seg_dist[count]:3d} 个连接 {bar}")
            L("")

        # E2E 延迟
        e2e = [c.e2e_seconds * 1000 for c in ok if c.e2e_seconds > 0]
        if e2e:
            L(f"  --- 端到端延迟 ({len(e2e)} 个连接) ---")
            L(_stat_line("E2E延迟", e2e))
            L("")

        # 每个分段明细（所有成功连接）
        L(f"  --- 分段明细 ---")
        L(f"  {'连接':>6} | {'分段':>4} | {'bg(ms)':>8} | {'ed(ms)':>8} | "
          f"{'音频(ms)':>8} | {'ASR(ms)':>9} | {'Total(ms)':>10} | {'ASR RTF':>8} | 文本")
        L(f"  {'-'*6} | {'-'*4} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*9} | {'-'*10} | {'-'*8} | {'-'*20}")

        for c in ok:
            for s in c.segments:
                rtf_str = f"{s.asr_rtf:.3f}" if s.asr_rtf > 0 else "N/A"
                L(f"  conn{c.conn_id:>3} | {s.seg_id:>4} | {s.bg_ms:>8} | {s.ed_ms:>8} | "
                  f"{s.audio_duration_ms:>8.0f} | {s.asr_ms:>9.1f} | {s.total_ms:>10.1f} | "
                  f"{rtf_str:>8} | {s.text}")

        L("")

    # ============================================================
    # 结尾
    # ============================================================
    L("=" * 100)
    L("报告结束")
    L("=" * 100)

    return "\n".join(lines)


# ============================================================
# 入口
# ============================================================

async def async_main(args):
    """异步主入口。"""
    audio_path = os.path.abspath(args.audio)
    print(f"Loading audio: {audio_path}")
    pcm_bytes = load_wav_pcm(audio_path)
    audio_duration = len(pcm_bytes) / 2 / 16000
    print(f"Audio duration: {audio_duration:.2f}s, PCM size: {len(pcm_bytes)} bytes")

    # 解析并发级别
    if args.levels:
        levels = [int(x.strip()) for x in args.levels.split(",")]
    else:
        levels = DEFAULT_LEVELS

    print(f"Concurrency levels: {levels}")
    print(f"Cooldown between levels: {args.cooldown}s")
    print(f"Target: {args.url}")
    print()

    chunk_bytes = args.chunk_samples * 2

    level_results: list[LevelResult] = []

    for i, level in enumerate(levels):
        print(f"{'='*60}")
        print(f"Running concurrency level {level} ({i+1}/{len(levels)})...")
        print(f"{'='*60}")

        lr = await run_level(
            url=args.url,
            pcm_bytes=pcm_bytes,
            concurrency=level,
            chunk_bytes=chunk_bytes,
            send_interval=args.send_interval,
            open_timeout=args.open_timeout,
            recv_timeout=args.recv_timeout,
            stagger_ms=args.stagger_ms,
        )
        level_results.append(lr)

        ok = sum(1 for c in lr.connections if c.success)
        fail = sum(1 for c in lr.connections if not c.success)
        total_segs = sum(len(c.segments) for c in lr.connections if c.success)
        all_asr = [s.asr_ms for c in lr.connections if c.success for s in c.segments if s.asr_ms > 0]
        asr_avg = f"{sum(all_asr)/len(all_asr):.1f}ms" if all_asr else "N/A"

        print(f"  Done in {lr.wall_time:.1f}s | OK={ok} Fail={fail} "
              f"Segments={total_segs} ASR_avg={asr_avg}")

        # 冷却
        if i < len(levels) - 1 and args.cooldown > 0:
            print(f"  Cooling down {args.cooldown}s...")
            await asyncio.sleep(args.cooldown)

    print()

    # 生成报告
    report = generate_full_report(level_results, audio_duration, args.url, audio_path)

    output_path = args.output
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(audio_path),
            f"asr_latency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to: {output_path}")
    print()

    # 打印汇总表
    print("Quick Summary:")
    for lr in level_results:
        ok = sum(1 for c in lr.connections if c.success)
        all_asr = [s.asr_ms for c in lr.connections if c.success for s in c.segments if s.asr_ms > 0]
        asr_avg = f"{sum(all_asr)/len(all_asr):.1f}" if all_asr else "N/A"
        asr_max = f"{max(all_asr):.1f}" if all_asr else "N/A"
        print(f"  C={lr.concurrency:>3}: OK={ok:>3}  wall={lr.wall_time:>7.1f}s  "
              f"ASR_avg={asr_avg:>8}ms  ASR_max={asr_max:>8}ms")


def main():
    parser = argparse.ArgumentParser(
        description="ASR 多并发时延分析测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认并发级别 (1,10,20,30,40,50,100,200,500)
  python test/asr_latency_profile.py --url ws://localhost:8856/tuling/ast/v3

  # 自定义并发级别
  python test/asr_latency_profile.py --levels 1,10,50

  # 指定输出文件
  python test/asr_latency_profile.py -o report.txt
        """,
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8856/tuling/ast/v3",
        help="WebSocket 服务地址 (默认: ws://localhost:8856/tuling/ast/v3)",
    )
    parser.add_argument(
        "--audio",
        default=os.path.join(os.path.dirname(__file__), "..", "120报警电话16k.wav"),
        help="测试音频文件路径 (PCM 16k 16bit mono WAV)",
    )
    parser.add_argument(
        "--levels",
        default="",
        help="并发级别（逗号分隔），默认: 1,10,20,30,40,50,100,200,500",
    )
    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=640,
        help="每次发送的样本数 (默认 640 = 40ms @ 16kHz)",
    )
    parser.add_argument(
        "--send-interval",
        type=float,
        default=0.04,
        help="发送间隔（秒），模拟实时 (默认 0.04 = 40ms)",
    )
    parser.add_argument(
        "--open-timeout",
        type=float,
        default=60.0,
        help="连接超时(秒) (默认 60)",
    )
    parser.add_argument(
        "--recv-timeout",
        type=float,
        default=300.0,
        help="接收超时(秒) (默认 300，高并发需要更长)",
    )
    parser.add_argument(
        "--stagger-ms",
        type=float,
        default=20,
        help="连接间错开毫秒数 (默认 20ms)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="并发级别间冷却时间(秒) (默认 5)",
    )
    parser.add_argument(
        "--output", "-o",
        default="",
        help="报告输出文件路径 (默认: 自动生成带时间戳的文件名)",
    )

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
