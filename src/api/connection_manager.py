"""
WebSocket 连接管理器 —— 并发控制与活跃连接注册表。
"""

from __future__ import annotations

from typing import Dict

from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """管理 WebSocket 连接的并发上限与生命周期。"""

    def __init__(self) -> None:
        self._max_connections: int = settings.MAX_CONNECTIONS
        self._active_count: int = 0
        self._active: Dict[str, str] = {}  # sid -> trace_id

    def try_acquire(self) -> bool:
        """
        尝试获取一个连接槽位（非阻塞）。

        Returns:
            True 表示成功获取，False 表示已满（应拒绝连接）。
        """
        if self._active_count >= self._max_connections:
            return False
        self._active_count += 1
        return True

    def register(self, sid: str, trace_id: str) -> None:
        """注册一个活跃连接。"""
        self._active[sid] = trace_id
        logger.info("Connection registered: sid=%s, trace_id=%s", sid, trace_id)

    def unregister(self, sid: str) -> None:
        """注销连接并释放槽位。"""
        self._active.pop(sid, None)
        self.release_slot()
        logger.info("Connection unregistered: sid=%s", sid)

    def release_slot(self) -> None:
        """释放一个连接槽位。"""
        self._active_count -= 1

    @property
    def active_count(self) -> int:
        """当前活跃连接数。"""
        return len(self._active)

    @property
    def active_connections(self) -> Dict[str, str]:
        """活跃连接快照。"""
        return dict(self._active)


# 全局单例
connection_manager = ConnectionManager()
