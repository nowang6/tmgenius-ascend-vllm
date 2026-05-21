# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 WebSocket 的高并发实时语音转录服务。处理管道：`音频流 → VAD 语音断句 → 异步 ASR (Qwen3-ASR-1.7B, Ascend NPU) → ITN 文本后处理 → JSON 结果推送`。

- **Python 版本**：3.11.14
- **ASR 推理**：vLLM-Ascend v0.18，由本服务通过 `main.py` 以子进程启动，HTTP 调用其 OpenAI 兼容接口
- **VAD 引擎**：TEN-VAD（C 原生库，每连接独立实例，通过 ctypes 调用）
- **ITN**：WeTextProcessing FST 模型，通过 8 实例 multiprocessing.Pool (spawn) 执行

## 常用命令

```bash
# 安装依赖
uv pip install .

# 安装开发依赖
uv pip install ".[dev]"

# 启动服务（同时启动 vLLM 子进程）
python main.py

# 运行测试
pytest test/ -v

# 运行单个测试
pytest test/ -k <test_name> -v

# 代码检查/格式化
ruff check src/
ruff format src/
mypy src/

# Docker 部署
docker-compose -f docker-compose.yaml up -d
```

## 架构

```
main.py                    # 入口：FastAPI lifespan 管理 vLLM 子进程、ITN Pool、ASR 客户端生命周期
├── src/core/
│   ├── config.py          # Settings 类，所有配置通过 os.getenv() 读取，代码内提供默认值
│   └── logging.py         # JSON 结构化日志（stdout），通过 ContextVar 自动注入 trace_id
├── src/api/
│   ├── websocket.py       # WebSocket 端点 /tuling/ast/v3，核心管道编排
│   ├── session.py         # ASRSession：每连接独立状态（seg_id、VAD 实例、热词、结果排序缓冲）
│   ├── connection_manager.py  # 并发上限控制（默认 64），超限返回 1013
│   ├── health.py          # /api/v1/{health,ready,connections}
│   └── metrics.py         # Prometheus 指标（连接数、延迟、队列深度、错误率、段数）
├── src/services/
│   ├── vad_service.py     # TenVADSession：每连接独立 TEN-VAD + 动态阈值断句状态机
│   ├── asr_service.py     # ASRService：httpx AsyncClient → vLLM /v1/chat/completions，含重试
│   └── itn_pool.py        # ITNPool：multiprocessing.Pool(spawn)，8 进程 eager init，通过 Queue 回传结果
├── src/models/schemas.py  # Pydantic 请求/响应模型（ClientMessage, ServerMessage 等）
└── src/utils/audio.py     # Base64 PCM/Opus 解码，通过 ctypes 调用 libopus.so.0
```

### 关键设计决策

- **VAD 每连接独立实例**：每个 WebSocket 连接创建独立 `TenVad` 实例（hop_size=640=40ms@16kHz），`process()` 通过 `asyncio.to_thread` 执行避免阻塞事件循环
- **ASR 异步后台处理**：VAD 触发断句后通过 `asyncio.create_task()` 启动后台 ASR+ITN 任务，**不阻塞**音频帧持续接收。通过 `ASRSession.send_lock`（`asyncio.Lock`）保证多个并发任务的 WebSocket 写入互斥
- **结果顺序保证**：`ASRSession._result_buffer`（按 seg_id 索引的字典）+ `_next_send_seg_id` 指针，确保长短句并发完成时推送顺序绝对递增
- **ITN 多进程池**：spawn 模式 8 进程，每进程预加载 `ITNProcessor` 单例，结果通过 `Manager().Queue()` 回传，主进程 `dispatcher` 线程分发给对应 waiter
- **vLLM 子进程管理**：`VLLMManager` 类管理 vLLM 的启停与健康检查，连续 3 次健康检查失败触发优雅关闭

### 配置关键项

所有配置通过环境变量覆盖，无配置文件。核心变量参见 `src/core/config.py:8-77`。对调试特别有用的：
- `LOG_LEVEL=DEBUG` 开启调试日志
- `VAD_THRESHOLD`、`VAD_PAUSE_MAX/MIN` 控制断句灵敏度
- `HOTWORDS` 设置服务端默认热词（逗号分隔），客户端可在每帧动态追加

### 模型文件位置

- `models/vad/ten-vad/` — TEN-VAD 原生库（libten_vad.so）+ ONNX 模型
- `models/itn/itn_wrapper.py` — ITN 模型包装器
- `weights/fst_itn_zh/` — ITN FST 模型文件
- `weights/Qwen3-ASR-1.7B/` — ASR 模型权重（通过 Docker Volume 挂载，不入镜像）
