"""
ASR 实时流式转录服务入口。

启动方式：
    python main.py
"""

import asyncio
import os
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from typing import ClassVar

import httpx

from fastapi import FastAPI

from src.core.config import settings

# ---- 将 config.py 的 WS Ping 配置注入环境变量，供 uvicorn CLI 启动时自动读取 ----
os.environ.setdefault("UVICORN_WS_PING_INTERVAL", str(int(settings.WS_PING_INTERVAL)))
os.environ.setdefault("UVICORN_WS_PING_TIMEOUT", str(int(settings.WS_PING_TIMEOUT)))

from src.api.health import router as health_router
from src.api.metrics import router as metrics_router
from src.api.websocket import asr_service, itn_pool, router as ws_router
from src.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class VLLMManager:
    """vLLM 子进程生命周期管理。"""

    process: ClassVar[subprocess.Popen | None] = None

    @classmethod
    def start(cls) -> bool:
        """启动 vLLM 推理服务子进程，阻塞等待就绪。"""
        if cls.process is not None and cls.process.poll() is None:
            logger.info("vLLM process already running")
            return True

        cmd = [
            "vllm", "serve", settings.VLLM_MODEL_PATH,
            "--served-model-name", settings.VLLM_MODEL_NAME,
            "--tensor-parallel-size", str(settings.VLLM_TENSOR_PARALLEL_SIZE),
            "--max-model-len", str(settings.VLLM_MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(settings.VLLM_GPU_MEMORY_UTILIZATION),
            "--port", str(settings.VLLM_PORT),
        ]

        if settings.VLLM_EXTRA_ARGS:
            cmd.extend(settings.VLLM_EXTRA_ARGS.split())

        logger.info("Starting vLLM: %s", " ".join(cmd))

        try:
            cls.process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
            )
            logger.info("vLLM started, PID: %d", cls.process.pid)
        except Exception:
            logger.exception("Failed to start vLLM")
            return False

        # 阻塞等待 vLLM API 就绪
        health_url = f"http://127.0.0.1:{settings.VLLM_PORT}/v1/models"
        timeout = settings.VLLM_STARTUP_TIMEOUT
        deadline = time.monotonic() + timeout

        logger.info("Waiting for vLLM at %s (timeout=%ds)...", health_url, timeout)
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                logger.error("vLLM exited with code %d", cls.process.returncode)
                return False
            try:
                r = httpx.get(health_url, timeout=2)
                if r.status_code == 200:
                    elapsed = timeout - (deadline - time.monotonic())
                    logger.info("vLLM is ready (took %.1fs)", elapsed)
                    return True
            except Exception:
                pass
            time.sleep(2)

        logger.error("vLLM did not become ready within %ds", timeout)
        cls.stop()
        return False

    @classmethod
    def stop(cls) -> None:
        """关闭 vLLM 子进程及其整个进程组。"""
        if cls.process is None:
            return
        try:
            pgid = os.getpgid(cls.process.pid)
        except OSError:
            cls.process = None
            return
        logger.info("Stopping vLLM process group (PGID %d)...", pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass
        try:
            cls.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            logger.warning("vLLM did not stop, killing process group...")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
            cls.process.wait()
        cls.process = None
        logger.info("vLLM stopped")

    @classmethod
    def is_alive(cls) -> bool:
        """检查 vLLM 进程是否存活。"""
        return cls.process is not None and cls.process.poll() is None


# 用于通知 lifespan 进行优雅关闭的事件
_shutdown_event: asyncio.Event | None = None

MAX_HEALTH_FAILURES = 3  # 连续失败次数阈值


async def _health_monitor() -> None:
    """后台持续监测 vLLM 健康，连续多次失败后通知优雅关闭。"""
    health_url = f"http://127.0.0.1:{settings.VLLM_PORT}/v1/models"
    interval = settings.VLLM_HEALTH_CHECK_INTERVAL
    consecutive_failures = 0

    async with httpx.AsyncClient(timeout=5) as client:
        while True:
            await asyncio.sleep(interval)

            # 进程级检查
            if not VLLMManager.is_alive():
                logger.critical("vLLM process died, triggering shutdown")
                if _shutdown_event:
                    _shutdown_event.set()
                return

            # HTTP 级检查
            try:
                # 添加 Connection: close 防止 vLLM(uvicorn) 的 5s keep-alive 机制导致连接断开抛出 RemoteProtocolError
                r = await client.get(health_url, headers={"Connection": "close"})
                if r.status_code == 200:
                    if consecutive_failures > 0:
                        logger.info("vLLM health recovered after %d failures", consecutive_failures)
                    consecutive_failures = 0
                    continue
                else:
                    logger.warning("vLLM health check returned %d", r.status_code)
            except Exception:
                logger.warning("vLLM health check request failed", exc_info=True)

            consecutive_failures += 1
            logger.warning(
                "vLLM health check failed (%d/%d)",
                consecutive_failures, MAX_HEALTH_FAILURES,
            )
            if consecutive_failures >= MAX_HEALTH_FAILURES:
                logger.critical(
                    "vLLM health check failed %d consecutive times, triggering shutdown",
                    MAX_HEALTH_FAILURES,
                )
                if _shutdown_event:
                    _shutdown_event.set()
                return


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动/关闭资源。"""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    # ---- 启动 ----
    logger.info(
        "Starting ASR service on %s:%d (max_conn=%d, ping_interval=%.0f, ping_timeout=%.0f)",
        settings.WS_HOST,
        settings.WS_PORT,
        settings.MAX_CONNECTIONS,
        settings.WS_PING_INTERVAL,
        settings.WS_PING_TIMEOUT,
    )

    # 1. 启动 vLLM（同步阻塞，跑在线程池里以免卡住事件循环）
    ok = await asyncio.to_thread(VLLMManager.start)
    if not ok:
        logger.critical("vLLM start failed, exiting")
        sys.exit(1)

    # 2. 启动 vLLM 健康监测后台任务
    monitor_task = asyncio.create_task(_health_monitor())

    # 3. 启动 shutdown 监听（健康检查失败时触发优雅退出）
    async def _watch_shutdown():
        await _shutdown_event.wait()
        logger.critical("Shutdown event received, stopping server...")
        # 给 uvicorn 发 SIGTERM 触发优雅关闭流程
        os.kill(os.getpid(), signal.SIGTERM)

    shutdown_watcher = asyncio.create_task(_watch_shutdown())

    # 4. ITN 多进程池（eager init）
    itn_pool.start()
    logger.info("ITN pool ready: %d workers", itn_pool.num_workers)

    # 5. ASR HTTP 客户端
    await asr_service.startup()

    logger.info("All services initialized")

    yield

    # ---- 关闭 ----
    logger.info("Shutting down ASR service...")
    monitor_task.cancel()
    shutdown_watcher.cancel()
    for t in (monitor_task, shutdown_watcher):
        try:
            await t
        except asyncio.CancelledError:
            pass
    await asr_service.shutdown()
    itn_pool.shutdown()
    VLLMManager.stop()
    logger.info("Shutdown complete")


app = FastAPI(
    title="ASR Real-time Streaming Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ws_router)
app.include_router(health_router)
app.include_router(metrics_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.WS_HOST,
        port=settings.WS_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        ws_ping_interval=settings.WS_PING_INTERVAL,
        ws_ping_timeout=settings.WS_PING_TIMEOUT,
    )
