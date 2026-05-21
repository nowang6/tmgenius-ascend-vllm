"""
健康检查与日志流端点。
"""

import asyncio

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from src.api.connection_manager import connection_manager
from src.core.logging import log_buffer

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health() -> dict:
    """服务进程存活检查。"""
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict:
    """模型加载就绪状态检查（K8s Readiness Probe）。"""
    # 复用全局 asr_service 实例（由 main.py lifespan 初始化）
    from src.api.websocket import asr_service

    available = await asr_service.is_available()
    if available:
        return {"status": "ready"}
    return {"status": "not_ready", "detail": "vLLM service unreachable"}


@router.get("/connections")
async def connections() -> dict:
    """当前活跃连接数统计。"""
    return {
        "active_connections": connection_manager.active_count,
        "details": connection_manager.active_connections,
    }


@router.get("/logs/stream")
async def logs_stream(
    backlog: int = Query(50, ge=0, le=2000, description="连接时先回放最近 N 条历史日志"),
) -> StreamingResponse:
    """SSE 实时日志流，类似 tail -f。"""

    async def event_generator():
        for entry in log_buffer.get_recent(backlog):
            yield f"data: {entry}\n\n"

        queue = log_buffer.subscribe()
        try:
            while True:
                try:
                    entry = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {entry}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            return
        finally:
            log_buffer.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )