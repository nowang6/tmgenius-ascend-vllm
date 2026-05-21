"""
Prometheus 指标定义与 /metrics 端点。
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()

# ---- 指标定义 ----
asr_connections_current = Gauge(
    "asr_connections_current",
    "当前活跃的 WebSocket 连接数",
)

asr_processing_latency_ms = Histogram(
    "asr_processing_latency_ms",
    "ASR 处理延迟（毫秒），含 vLLM HTTP 调用耗时",
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000],
)

asr_segments_total = Counter(
    "asr_segments_total",
    "已处理的语音段总数",
)

asr_errors_total = Counter(
    "asr_errors_total",
    "ASR 处理错误总数",
    ["error_type"],
)

asr_queue_depth = Gauge(
    "asr_queue_depth",
    "ITN 多进程池待处理任务数（队列深度）",
)


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus 指标暴露端点。"""
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
