"""
ITN 多进程池 —— 基于 multiprocessing.Pool (spawn) 的生产实现。

架构说明：
  - 使用 spawn 模式创建 N 个 worker 进程
  - 每个进程启动时预加载 ITNProcessor 单例（避免运行时延迟）
  - 请求通过 Pool.apply_async 分发（Pool 内部队列自动负载均衡）
  - 结果通过 Manager().Queue() 跨进程安全回传
  - 主进程端通过 dispatcher 线程轮询结果队列，分发给对应的 waiter

生命周期：
  - 应用启动时调用 start()，预创建所有进程并预热 ITN 模型（eager init）
  - 应用关闭时调用 shutdown()，终止所有进程

"""

from __future__ import annotations

import asyncio
import concurrent.futures
import multiprocessing as mp
import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from src.api.metrics import asr_queue_depth
from src.core.config import settings
from src.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================
# Worker 进程内全局变量
# ============================================================

_ITN_PROCESSOR: Any = None
_RESULT_QUEUE: Any = None


def _safe_qsize(queue_obj: Any) -> Optional[int]:
    """Best-effort 读取队列长度；不支持时返回 None。"""
    try:
        return int(queue_obj.qsize())
    except Exception:
        return None


def _init_itn_worker(result_queue: Any) -> None:
    """Pool initializer：子进程启动时执行一次，预加载 ITN 模型。"""
    global _ITN_PROCESSOR, _RESULT_QUEUE
    _RESULT_QUEUE = result_queue

    import importlib.util
    import os

    models_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "weights", "itn")
    )
    wrapper_path = os.path.join(models_dir, "itn_wrapper.py")
    spec = importlib.util.spec_from_file_location("itn_wrapper", wrapper_path)
    itn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(itn_module)

    _ITN_PROCESSOR = itn_module.ITNProcessor()
    logger.info(
        "ITN worker process initialized: pid=%d, model_dir=%s",
        mp.current_process().pid,
        models_dir,
    )


def _itn_worker_task(task: dict) -> None:
    """
    Worker 执行函数：执行 ITN normalize。

    结果通过 _RESULT_QUEUE 回传。
    """
    task_id = task["task_id"]
    text = task["text"]

    try:
        if not text or not text.strip():
            result_text = ""
        else:
            result_text = _ITN_PROCESSOR.process(text)

        _RESULT_QUEUE.put(
            {"task_id": task_id, "ok": True, "payload": {"text": result_text}}
        )
    except Exception as exc:
        _RESULT_QUEUE.put({"task_id": task_id, "ok": False, "error": repr(exc)})


# ============================================================
# ITNPool —— 生产级多进程池
# ============================================================


@dataclass
class _ITNWorkerRuntime:
    pool: Any
    result_queue: Any
    pending: dict[str, queue.Queue] = field(default_factory=dict)
    pending_lock: threading.Lock = field(default_factory=threading.Lock)
    dispatcher: Optional[threading.Thread] = None
    running: threading.Event = field(default_factory=threading.Event)


class ITNPool:
    """
    ITN 多进程池。

    - num_workers 个进程，每个进程预加载一个 ITNProcessor 实例
    - 请求通过 Pool 内部队列自动分发（负载均衡）
    - 结果通过 Manager().Queue() 回传
    """

    def __init__(self, num_workers: int | None = None):
        self._num_workers = num_workers or settings.ITN_WORKERS

        if self._num_workers <= 0:
            raise ValueError("num_workers must be > 0")

        self._ctx = mp.get_context("spawn")
        self._manager: Any = None
        self._runtime: Optional[_ITNWorkerRuntime] = None
        self._monitor_interval_sec = max(0.0, settings.MP_QUEUE_LOG_INTERVAL_SEC)
        self._monitor_running = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        # 独立线程池，避免与 VAD / ASR 编码共用默认 executor
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._num_workers * 4,
            thread_name_prefix="itn-submit",
        )

    def start(self) -> None:
        """
        启动所有 worker 进程（应用启动时调用，eager init）。

        所有进程在 initializer 中预加载 ITN 模型，启动完成即就绪。
        """
        self._manager = self._ctx.Manager()
        result_q = self._manager.Queue()

        pool = self._ctx.Pool(
            processes=self._num_workers,
            initializer=_init_itn_worker,
            initargs=(result_q,),
        )

        self._runtime = _ITNWorkerRuntime(
            pool=pool,
            result_queue=result_q,
        )
        self._runtime.running.set()
        self._runtime.dispatcher = threading.Thread(
            target=self._dispatch_results_loop,
            args=(self._runtime,),
            daemon=True,
            name="itn-dispatch",
        )
        self._runtime.dispatcher.start()

        if self._monitor_interval_sec > 0:
            self._monitor_running.clear()
            self._monitor_thread = threading.Thread(
                target=self._log_queue_stats_loop,
                daemon=True,
                name="itn-queue-monitor",
            )
            self._monitor_thread.start()

        logger.info("ITN pool started: %d workers", self._num_workers)

    def shutdown(self) -> None:
        """终止所有 worker 进程并释放资源。"""
        self._monitor_running.set()
        if self._runtime:
            self._runtime.running.clear()
            try:
                self._runtime.pool.close()
                self._runtime.pool.join(timeout=5)
            except Exception:
                pass
            try:
                self._runtime.pool.terminate()
                self._runtime.pool.join()
            except Exception:
                pass
        try:
            if self._manager:
                self._manager.shutdown()
        except Exception:
            pass
        self._monitor_thread = None
        self._runtime = None
        logger.info("ITN pool shutdown complete")

    # ---- 结果分发循环 ----

    def _dispatch_results_loop(self, runtime: _ITNWorkerRuntime) -> None:
        """后台线程：轮询 result_queue，将结果分发给对应的 waiter。"""
        while runtime.running.is_set():
            try:
                result = runtime.result_queue.get(timeout=0.2)
            except Exception:
                continue
            task_id = result.get("task_id")
            if not task_id:
                continue
            with runtime.pending_lock:
                waiter = runtime.pending.pop(task_id, None)
            if waiter is not None:
                waiter.put(result)

    def _log_queue_stats_loop(self) -> None:
        """后台线程：定期打印队列与待回包任务长度，并更新 Prometheus 指标。"""
        while not self._monitor_running.wait(timeout=self._monitor_interval_sec):
            runtime = self._runtime
            if runtime is None:
                continue

            with runtime.pending_lock:
                pending_size = len(runtime.pending)

            result_qsize = _safe_qsize(runtime.result_queue)
            result_qsize_text = (
                str(result_qsize) if result_qsize is not None else "unknown"
            )

            asr_queue_depth.set(pending_size)

            logger.info(
                "ITN queue stats: pending=%d result_queue=%s workers=%d",
                pending_size,
                result_qsize_text,
                self._num_workers,
            )

    # ---- 同步提交 ----

    def _submit(self, text: str, timeout_sec: float = 30.0) -> str:
        """提交 ITN 任务并等待结果（同步阻塞）。"""
        assert self._runtime is not None, "ITNPool not started"

        task_id = uuid.uuid4().hex
        task = {"task_id": task_id, "text": text}

        waiter: queue.Queue = queue.Queue(maxsize=1)
        with self._runtime.pending_lock:
            self._runtime.pending[task_id] = waiter

        self._runtime.pool.apply_async(_itn_worker_task, args=(task,))

        try:
            result = waiter.get(timeout=timeout_sec)
        except Exception as exc:
            with self._runtime.pending_lock:
                self._runtime.pending.pop(task_id, None)
            raise TimeoutError(f"ITN task timeout: {task_id}") from exc

        if not result.get("ok"):
            raise RuntimeError(result.get("error", "unknown ITN worker error"))
        return result["payload"]["text"]

    # ---- 异步接口（供 WebSocket handler 调用） ----

    async def normalize(self, text: str) -> str:
        """
        异步执行逆正则化。

        Args:
            text: ASR 原始输出文本

        Returns:
            逆正则化后的标准化文本
        """
        if not text or not text.strip():
            return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._submit, text)

    @property
    def num_workers(self) -> int:
        return self._num_workers
