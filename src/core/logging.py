"""
JSON 结构化日志配置。

- 输出至 stdout
- 每条日志自动注入 trace_id 字段
- 不记录 Base64 音频原文
- 内存环形缓冲区供 HTTP 流式查询
"""

import asyncio
import collections
import logging
import json
import sys
from contextvars import ContextVar
from threading import Lock

from src.core.config import settings

# 用于在异步上下文中传递 trace_id
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


class JSONFormatter(logging.Formatter):
    """将日志格式化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get("-"),
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class InMemoryLogHandler(logging.Handler):
    """将日志写入内存环形缓冲区，并通知所有 SSE 订阅者。"""

    def __init__(self, capacity: int = 2000):
        super().__init__()
        self._buffer: collections.deque[str] = collections.deque(maxlen=capacity)
        self._lock = Lock()
        self._subscribers: set[asyncio.Queue[str]] = set()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg)
            for q in list(self._subscribers):
                try:
                    q.put_nowait(msg)
                except asyncio.QueueFull:
                    pass
        except Exception:
            self.handleError(record)

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    def get_recent(self, n: int = 50) -> list[str]:
        with self._lock:
            items = list(self._buffer)
        return items[-n:]


log_buffer = InMemoryLogHandler(capacity=2000)


def setup_logging() -> None:
    """初始化全局日志配置。"""
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL.upper())

    # 清除已有 handler，避免重复
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    root.addHandler(handler)

    log_buffer.setFormatter(JSONFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    root.addHandler(log_buffer)

    # 降低第三方库日志级别
    for name in ("uvicorn", "uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取带模块名的 logger。"""
    return logging.getLogger(name)