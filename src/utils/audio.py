"""
音频处理工具函数。
"""

import base64
import ctypes
from typing import Optional

import numpy as np


SAMPLE_RATE = 16000  # 16kHz
SAMPLE_WIDTH = 2  # 16bit = 2 bytes

# ============================================================
# libopus 绑定（系统 libopus.so.0，无额外 Python 依赖）
# ============================================================

_opus_lib = ctypes.cdll.LoadLibrary("libopus.so.0")

_opus_lib.opus_decoder_create.restype = ctypes.c_void_p
_opus_lib.opus_decoder_create.argtypes = [
    ctypes.c_int,  # Fs (采样率)
    ctypes.c_int,  # channels
    ctypes.POINTER(ctypes.c_int),  # error
]

_opus_lib.opus_decoder_destroy.argtypes = [ctypes.c_void_p]

_opus_lib.opus_decode.restype = ctypes.c_int
_opus_lib.opus_decode.argtypes = [
    ctypes.c_void_p,  # decoder
    ctypes.POINTER(ctypes.c_ubyte),  # data (opus 包，可为 NULL 触发 PLC)
    ctypes.c_int,  # len (包字节数)
    ctypes.POINTER(ctypes.c_short),  # pcm (int16 输出缓冲)
    ctypes.c_int,  # frame_size (最大采样数)
    ctypes.c_int,  # decode_fec (0=正常, 1=FEC)
]

_opus_lib.opus_packet_get_nb_samples.restype = ctypes.c_int
_opus_lib.opus_packet_get_nb_samples.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),  # packet
    ctypes.c_int,  # len
    ctypes.c_int,  # Fs
]

_opus_lib.opus_strerror.restype = ctypes.c_char_p
_opus_lib.opus_strerror.argtypes = [ctypes.c_int]

# Opus 最大帧长：16kHz 下 120ms = 1920 采样，1 通道
_OPUS_MAX_FRAME_SAMPLES = 1920


def _check_opus_error(code: int) -> None:
    if code < 0:
        err_msg = _opus_lib.opus_strerror(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"Opus decode error {code}: {err_msg}")


class OpusDecoder:
    """libopus 解码器的 ctypes 封装。

    解码器是有状态的（支持 PLC），同一连接复用同一个实例。
    """

    def __init__(self, sr: int = 16000, channels: int = 1) -> None:
        err = ctypes.c_int(0)
        self._handle: Optional[int] = _opus_lib.opus_decoder_create(
            sr, channels, ctypes.byref(err)
        )
        _check_opus_error(err.value)
        self._sr = sr

    def decode(self, opus_packet: bytes) -> np.ndarray:
        """解码单个 opus 包，返回 int16 numpy 数组。"""
        if self._handle is None:
            raise RuntimeError("OpusDecoder already closed")

        pkt_len = len(opus_packet)
        pkt_buf = (ctypes.c_ubyte * pkt_len).from_buffer_copy(opus_packet)

        # 获取包内采样数
        nb_samples = _opus_lib.opus_packet_get_nb_samples(pkt_buf, pkt_len, self._sr)
        if nb_samples <= 0:
            _check_opus_error(nb_samples)

        # 分配输出缓冲并解码
        pcm_buf = (ctypes.c_short * nb_samples)()
        ret = _opus_lib.opus_decode(
            self._handle,
            pkt_buf,
            pkt_len,
            pcm_buf,
            nb_samples,
            0,  # decode_fec=0
        )
        _check_opus_error(ret)

        return np.ctypeslib.as_array(pcm_buf).copy()

    def close(self) -> None:
        if self._handle is not None:
            _opus_lib.opus_decoder_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> "OpusDecoder":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def decode_base64_opus(b64_data: str, decoder: OpusDecoder) -> np.ndarray:
    """将 Base64 编码的 Opus 数据解码为 int16 numpy 数组。"""
    raw_bytes = base64.b64decode(b64_data)
    return decoder.decode(raw_bytes)


# ============================================================
# PCM 解码
# ============================================================


def decode_base64_pcm(b64_data: str) -> np.ndarray:
    """将 Base64 编码的 PCM 16k/16bit 数据解码为 int16 numpy 数组。"""
    raw_bytes = base64.b64decode(b64_data)
    return np.frombuffer(raw_bytes, dtype=np.int16).copy()


def int16_to_float32(audio: np.ndarray) -> np.ndarray:
    """int16 → float32（归一化到 [-1, 1]）。"""
    return audio.astype(np.float32) / 32768.0


def samples_to_ms(samples: int, sr: int = SAMPLE_RATE) -> int:
    """采样数 → 毫秒。"""
    return int(samples * 1000 / sr)


def samples_to_cs(samples: int, sr: int = SAMPLE_RATE) -> int:
    """采样数 → 厘秒（10ms 为 1 厘秒）。"""
    return int(samples * 100 / sr)
