"""
Opus 编码模拟客户端 —— 用于测试服务端 encoding=opus 解码通路。

用法：
    python test/opus_test_client.py <wav_file> [--url ws://host:port/tuling/ast/v3] [--frame-ms 20]

示例：
    python test/opus_test_client.py test/data/sample.wav
    python test/opus_test_client.py test/data/sample.wav --url ws://127.0.0.1:15003/tuling/ast/v3 --frame-ms 40

要求：
    - WAV 文件为 16kHz 单声道（脚本会自动重采样，但建议预先准备）
    - 服务端需先启动
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import json
import os
import sys
import time
import uuid

import numpy as np
import soundfile as sf
import websockets

# ============================================================
# libopus 编码器 ctypes 绑定
# ============================================================

_opus_lib = ctypes.cdll.LoadLibrary("libopus.so.0")

_opus_lib.opus_encoder_create.restype = ctypes.c_void_p
_opus_lib.opus_encoder_create.argtypes = [
    ctypes.c_int,  # Fs
    ctypes.c_int,  # channels
    ctypes.c_int,  # application (OPUS_APPLICATION_VOIP=2048)
    ctypes.POINTER(ctypes.c_int),  # error
]

_opus_lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]

_opus_lib.opus_encode.restype = ctypes.c_int
_opus_lib.opus_encode.argtypes = [
    ctypes.c_void_p,  # encoder
    ctypes.POINTER(ctypes.c_short),  # pcm (int16)
    ctypes.c_int,  # frame_size
    ctypes.POINTER(ctypes.c_ubyte),  # data (输出缓冲)
    ctypes.c_int,  # max_data_bytes
]

_opus_lib.opus_strerror.restype = ctypes.c_char_p
_opus_lib.opus_strerror.argtypes = [ctypes.c_int]

OPUS_APPLICATION_VOIP = 2048
OPUS_MAX_PACKET = 4000


def _check_opus_error(code: int) -> None:
    if code < 0:
        err_msg = _opus_lib.opus_strerror(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"Opus error {code}: {err_msg}")


class OpusEncoder:
    """libopus 编码器的轻量封装。"""

    def __init__(self, sr: int = 16000, channels: int = 1) -> None:
        err = ctypes.c_int(0)
        self._handle = _opus_lib.opus_encoder_create(
            sr, channels, OPUS_APPLICATION_VOIP, ctypes.byref(err)
        )
        _check_opus_error(err.value)
        self._sr = sr

    def encode(self, pcm_int16: np.ndarray) -> bytes:
        """编码一段 int16 PCM 为一个 opus 包。"""
        frame_size = len(pcm_int16)
        pcm_buf = (ctypes.c_short * frame_size).from_buffer_copy(pcm_int16)
        out_buf = (ctypes.c_ubyte * OPUS_MAX_PACKET)()
        ret = _opus_lib.opus_encode(
            self._handle, pcm_buf, frame_size, out_buf, OPUS_MAX_PACKET
        )
        _check_opus_error(ret)
        return bytes(out_buf[:ret])

    def close(self) -> None:
        if self._handle is not None:
            _opus_lib.opus_encoder_destroy(self._handle)
            self._handle = None


# ============================================================
# WebSocket 客户端逻辑
# ============================================================


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """加载 WAV 文件并转为 16kHz 单声道 int16。"""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    # 简单重采样（实际项目建议用 torchaudio 或 scipy）
    if sr != target_sr:
        import math
        ratio = target_sr / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sr
    # 转 int16
    if audio.dtype != np.int16:
        # 假设浮点在 [-1, 1]
        audio = (audio * 32767).astype(np.int16)
    return audio


async def run_opus_client(
    url: str,
    wav_path: str,
    frame_ms: int = 20,
    trace_id: str | None = None,
) -> None:
    """通过 WebSocket 发送 opus 编码音频。"""

    if trace_id is None:
        trace_id = str(uuid.uuid4())

    audio = load_wav(wav_path, target_sr=16000)
    frame_samples = int(16000 * frame_ms / 1000)
    total_frames = (len(audio) + frame_samples - 1) // frame_samples

    print(f"Audio: {wav_path}")
    print(f"Duration: {len(audio) / 16000:.2f}s, frames: {total_frames} ({frame_ms}ms each)")
    print(f"Server: {url}")
    print(f"Trace ID: {trace_id}")
    print()

    encoder = OpusEncoder(sr=16000)
    try:
        async with websockets.connect(url, ping_interval=10, ping_timeout=5) as ws:
            # ---- 握手帧 (status=0) ----
            handshake = {
                "header": {
                    "traceId": trace_id,
                    "appId": "opus-test",
                    "bizId": "test-biz",
                    "status": 0,
                }
            }
            await ws.send(json.dumps(handshake, ensure_ascii=False))
            resp = json.loads(await ws.recv())
            print(f"Handshake response: code={resp['header']['code']}, sid={resp['header']['sid']}")
            if resp["header"]["code"] != 0:
                print(f"ERROR: handshake failed: {resp['header']['message']}")
                return

            sid = resp["header"]["sid"]

            # ---- 逐帧发送音频 (status=1, encoding=opus) ----
            t_start = time.monotonic()
            audio_offset = 0

            for i in range(total_frames):
                chunk = audio[audio_offset : audio_offset + frame_samples]
                audio_offset += frame_samples

                # 不足一帧时补零
                if len(chunk) < frame_samples:
                    chunk = np.pad(chunk, (0, frame_samples - len(chunk)))

                # 编码为 opus
                opus_pkt = encoder.encode(chunk)
                b64 = base64.b64encode(opus_pkt).decode("ascii")

                msg = {
                    "header": {
                        "traceId": trace_id,
                        "appId": "opus-test",
                        "bizId": "test-biz",
                        "status": 1,
                    },
                    "payload": {
                        "audio": {
                            "audio": b64,
                            "encoding": "opus",
                        }
                    },
                }
                await ws.send(json.dumps(msg, ensure_ascii=False))

                # 接收可能的结果推送
                try:
                    while True:
                        resp_raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                        resp = json.loads(resp_raw)
                        hdr = resp["header"]
                        if hdr["status"] == 1:
                            result = resp["payload"]["result"]
                            text = result.get("ws", [{}])[0].get("cw", [{}])[0].get("w", "")
                            print(f"  [seg {result['segId']}] {text}")
                except asyncio.TimeoutError:
                    pass

                if (i + 1) % 50 == 0:
                    print(f"  ... sent {i + 1}/{total_frames} frames")

            # ---- 结束帧 (status=2) ----
            end_msg = {
                "header": {
                    "traceId": trace_id,
                    "appId": "opus-test",
                    "bizId": "test-biz",
                    "status": 2,
                }
            }
            await ws.send(json.dumps(end_msg, ensure_ascii=False))

            # ---- 接收剩余结果和终态 ----
            print("\nFinal results:")
            while True:
                try:
                    resp_raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    resp = json.loads(resp_raw)
                    hdr = resp["header"]
                    if hdr["status"] == 2:
                        print(f"  [END] code={hdr['code']}, sid={hdr['sid']}")
                        break
                    elif hdr["status"] == 1:
                        result = resp["payload"]["result"]
                        text = result.get("ws", [{}])[0].get("cw", [{}])[0].get("w", "")
                        print(f"  [seg {result['segId']}] {text}")
                except asyncio.TimeoutError:
                    print("  (timeout waiting for final response)")
                    break

            elapsed = time.monotonic() - t_start
            print(f"\nDone. Audio: {len(audio) / 16000:.1f}s, elapsed: {elapsed:.1f}s, RTF: {elapsed / (len(audio) / 16000):.2f}")
    finally:
        encoder.close()


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Opus encoding WebSocket test client for ASR service"
    )
    parser.add_argument(
        "wav", metavar="WAV_FILE",
        help="Path to a 16kHz mono WAV file",
    )
    parser.add_argument(
        "--url", default="ws://127.0.0.1:15003/tuling/ast/v3",
        help="WebSocket endpoint URL (default: ws://127.0.0.1:15003/tuling/ast/v3)",
    )
    parser.add_argument(
        "--frame-ms", type=int, default=20,
        help="Opus frame duration in ms (default: 20)",
    )
    parser.add_argument(
        "--trace-id",
        help="Trace ID (auto-generated if not specified)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.wav):
        print(f"ERROR: file not found: {args.wav}")
        sys.exit(1)

    asyncio.run(run_opus_client(args.url, args.wav, args.frame_ms, args.trace_id))


if __name__ == "__main__":
    main()
