#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
import asyncio
import time
import subprocess
import signal
import sys
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
from openai import AsyncOpenAI, APITimeoutError, APIConnectionError, APIStatusError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 配置参数 ====================
# vLLM 配置
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"
VLLM_OPENAI_BASE_URL = f"{VLLM_BASE_URL}/v1"

# vLLM 启动参数
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/HY-MT1.5-7B")
MODEL_NAME = os.environ.get("MODEL_NAME", "HY-MT1.5")
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8"))
VLLM_EXTRA_ARGS = os.environ.get("VLLM_EXTRA_ARGS", "")

# 代理服务配置
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8858"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "120"))
ENABLE_TEXT_SPLIT = os.environ.get("ENABLE_TEXT_SPLIT", "true").lower() == "true"
VLLM_STARTUP_TIMEOUT = int(os.environ.get("VLLM_STARTUP_TIMEOUT", "300"))

# 翻译提示词模板 (支持 {source_language_name} 和 {target_language_name} 占位符)
TRANSLATION_PROMPT_TEMPLATE = os.environ.get(
    "TRANSLATION_PROMPT_TEMPLATE",
    "你是翻译助手，将{source_language_name}翻译成{target_language_name}。只输出翻译结果，保持原文语气。若是符号或无意义文本则输出空。"
)

# 语言映射
LANGUAGE_MAP = {
    "cn": "中文", "en": "英语", "fr": "法语",
    "pt": "葡萄牙语", "es": "西班牙语", "ja": "日语", "tr": "土耳其语",
    "ru": "俄语", "ar": "阿拉伯语", "ko": "韩语", "th": "泰语",
    "it": "意大利语", "de": "德语", "vi": "越南语", "ms": "马来语",
    "id": "印尼语", "tl": "菲律宾语", "hi": "印地语", 
    "pl": "波兰语", "cs": "捷克语", "nl": "荷兰语", "km": "高棉语",
    "my": "缅甸语", "fa": "波斯语", "gu": "古吉拉特语", "ur": "乌尔都语",
    "te": "泰卢固语", "mr": "马拉地语", "he": "希伯来语", "bn": "孟加拉语",
    "ta": "泰米尔语", "uk": "乌克兰语", "bo": "藏语", "kk": "哈萨克语",
    "mn": "蒙古语", "ug": "维吾尔语", "yu": "粤语",
}

# 断句标点
SENTENCE_END = re.compile(r'([。！？!?…]+|\.(?=\s|$))')

# ==================== 自定义异常 ====================

class TranslationError(Exception):
    """翻译异常，携带 HTTP 状态码"""
    def __init__(self, message: str, status_code: int = 500, text: str = ""):
        self.message = message
        self.status_code = status_code
        self.text = text  # 原始文本（用于日志）
        super().__init__(self.message)

# ==================== 全局状态 ====================

class TranslationState:
    """翻译代理服务全局状态"""
    openai_client: Optional[AsyncOpenAI] = None  # OpenAI 客户端（翻译用）
    http_client: Optional[httpx.AsyncClient] = None  # HTTP 客户端（健康检查/metrics 代理用）
    vllm_process: Optional[subprocess.Popen] = None  # vLLM 子进程
    total_requests = 0
    total_texts = 0
    total_time_ms = 0.0

    @classmethod
    async def init(cls):
        """初始化 OpenAI 客户端和 HTTP 客户端"""
        if cls.openai_client is None:
            cls.openai_client = AsyncOpenAI(
                base_url=VLLM_OPENAI_BASE_URL,
                api_key="EMPTY",
                timeout=REQUEST_TIMEOUT
            )
            cls.http_client = httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT))
            logger.info(f"初始化: OpenAI SDK -> {VLLM_OPENAI_BASE_URL}")

    @classmethod
    async def close(cls):
        """清理资源"""
        if cls.openai_client:
            await cls.openai_client.close()
            cls.openai_client = None
        if cls.http_client:
            await cls.http_client.aclose()
            cls.http_client = None
        cls.stop_vllm()

    @classmethod
    def start_vllm(cls) -> bool:
        """启动 vLLM server 子进程"""
        if cls.vllm_process is not None and cls.vllm_process.poll() is None:
            logger.info("vLLM 进程已在运行")
            return True
        
        # 构建 vLLM 启动命令
        cmd = [
            "vllm", "serve", MODEL_PATH,
            "--served-model-name", MODEL_NAME,
            "--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
            "--port", str(VLLM_PORT),
        ]
        
        # 添加额外参数
        if VLLM_EXTRA_ARGS:
            cmd.extend(VLLM_EXTRA_ARGS.split())
        
        logger.info(f"启动 vLLM: {' '.join(cmd)}")
        
        try:
            # stdout/stderr 不重定向，让 vLLM 输出直接打印到控制台
            cls.vllm_process = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid  # 创建新进程组，方便统一管理
            )
            logger.info(f"vLLM 进程已启动, PID: {cls.vllm_process.pid}")
            return True
        except Exception as e:
            logger.error(f"启动 vLLM 失败: {e}")
            return False

    @classmethod
    def stop_vllm(cls):
        """停止 vLLM server 子进程"""
        if cls.vllm_process is not None:
            logger.info(f"停止 vLLM 进程, PID: {cls.vllm_process.pid}")
            try:
                os.killpg(os.getpgid(cls.vllm_process.pid), signal.SIGTERM)
                cls.vllm_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("vLLM 进程未响应 SIGTERM，发送 SIGKILL")
                os.killpg(os.getpgid(cls.vllm_process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"停止 vLLM 进程异常: {e}")
            finally:
                cls.vllm_process = None

    @classmethod
    def is_vllm_running(cls) -> bool:
        """检查 vLLM 进程是否在运行"""
        if cls.vllm_process is None:
            return False
        return cls.vllm_process.poll() is None

# ==================== 生命周期管理 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理 - 先启动 vLLM，成功后再启动代理"""
    logger.info("=" * 50)
    logger.info("=" * 50)
    logger.info(f"模型路径: {MODEL_PATH} | 模型名称: {MODEL_NAME}")
    logger.info(f"vLLM 端口: {VLLM_PORT} | 代理端口: {PROXY_PORT}")
    logger.info(f"OpenAI SDK -> {VLLM_OPENAI_BASE_URL}")
    logger.info(f"句子切分: {'启用' if ENABLE_TEXT_SPLIT else '禁用'}")
    
    # 步骤 1: 启动 vLLM server
    if not TranslationState.start_vllm():
        logger.error("vLLM 启动失败，服务退出")
        sys.exit(1)
    
    # 步骤 2: 等待 vLLM 就绪
    if not await wait_vllm(timeout=VLLM_STARTUP_TIMEOUT):
        logger.error("vLLM 启动超时，服务退出")
        TranslationState.stop_vllm()
        sys.exit(1)
    
    # 步骤 3: 初始化代理服务
    await TranslationState.init()
    logger.info(f"服务已启动: http://0.0.0.0:{PROXY_PORT}/turing/itrans/v3/func")
    
    yield  # 服务运行中
    
    await TranslationState.close()
    logger.info("服务停止")

# 创建 FastAPI 应用
app = FastAPI(
    title="Translation Proxy",
    version="4.0",
    lifespan=lifespan
)

# ==================== 工具函数 ====================
def parse_lang_type(type_str: str) -> Tuple[str, str]:
    """解析语言类型: 'cnen' -> ('cn', 'en')"""
    s = type_str.lower()
    if len(s) == 4:
        return s[:2], s[2:]
    for sep in ["2", "to", "_", "-"]:
        if sep in s:
            parts = s.split(sep, 1)
            return parts[0], parts[1]
    for i in range(2, min(8, len(s))):
        if s[:i] in LANGUAGE_MAP and s[i:] in LANGUAGE_MAP:
            return s[:i], s[i:]
    return s[:2], s[2:] if len(s) >= 4 else (s, s)

# 增加单元测试，确保切分正确
def split_sentences(text: str) -> List[str]:
    """按断句标点切分文本"""
    if not text or not text.strip():
        return [text] if text else []
    parts = SENTENCE_END.split(text)
    sentences, i = [], 0
    while i < len(parts):
        if i + 1 < len(parts) and SENTENCE_END.match(parts[i + 1]):
            sent = (parts[i] + parts[i + 1]).strip()
            i += 2
        else:
            sent = parts[i].strip()
            i += 1
        if sent:
            sentences.append(sent)
    return sentences

# ==================== 翻译核心 ====================
async def translate_single(text: str, source_language_name: str, target_language_name: str) -> str:
    """
    翻译单条文本（使用 OpenAI SDK）
    
    Raises:
        TranslationError: 后端异常时抛出，携带 HTTP 状态码
    """
    prompt = TRANSLATION_PROMPT_TEMPLATE.format(
        source_language_name=source_language_name,
        target_language_name=target_language_name
    )
    try:
        completion = await TranslationState.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            top_p=0.6,
            max_tokens=1024,
            extra_body={
                "top_k": 20,
                "repetition_penalty": 1.05,
            }
        )
        result = completion.choices[0].message.content
        return result.strip().strip("{}").strip() if result else ""
    except TranslationError:
        raise  # 直接向上传递
    except APITimeoutError:
        error_msg = f"翻译超时: {text[:50]}..."
        logger.error(error_msg)
        raise TranslationError(error_msg, status_code=504, text=text)
    except APIConnectionError:
        error_msg = f"连接vLLM失败: {text[:50]}..."
        logger.error(error_msg)
        raise TranslationError(error_msg, status_code=502, text=text)
    except APIStatusError as e:
        error_msg = f"vLLM错误 {e.status_code}: {str(e)[:200]}"
        logger.error(error_msg)
        raise TranslationError(error_msg, status_code=e.status_code, text=text)
    except Exception as e:
        error_msg = f"翻译失败: {e}, 文本: {text[:50]}..."
        logger.error(error_msg)
        raise TranslationError(error_msg, status_code=500, text=text)


async def translate_batch(texts: List[str], source_language_name: str, target_language_name: str) -> Tuple[List[str], Optional[TranslationError]]:
    """
    翻译多条文本
    
    Returns:
        (翻译结果列表, 第一个遇到的错误或None)
        - 单条失败不影响其他条目，失败的返回空字符串
        - 如果有任何错误，返回第一个错误供调用方决定是否返回错误状态码
    """
    if not texts:
        return [], None
    
    tasks = [translate_single(t, source_language_name, target_language_name) for t in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    translations = []
    first_error: Optional[TranslationError] = None
    
    for r in results:
        if isinstance(r, str):
            translations.append(r)
        elif isinstance(r, TranslationError):
            translations.append("")  # 失败的返回空字符串
            if first_error is None:
                first_error = r
        else:
            translations.append("")
            if first_error is None:
                first_error = TranslationError(str(r), status_code=500)
    
    return translations, first_error

async def wait_vllm(timeout: int = 300) -> bool:
    """等待 vLLM 就绪，同时监控进程状态"""
    start = time.time()
    async with httpx.AsyncClient(timeout=5) as client:
        while time.time() - start < timeout:
            # 检查 vLLM 进程是否还在运行
            if not TranslationState.is_vllm_running():
                logger.error("vLLM 进程已退出")
                return False
            
            try:
                resp = await client.get(f"{VLLM_BASE_URL}/v1/models")
                if resp.status_code == 200:
                    logger.info("vLLM 已就绪")
                    return True
            except: 
                pass
            logger.info(f"等待 vLLM... ({int(time.time()-start)}s)")
            await asyncio.sleep(5)
    logger.error("vLLM 启动超时")
    return False


async def check_vllm_health() -> bool:
    """检查 vLLM 服务健康状态"""
    # 检查进程是否在运行
    if not TranslationState.is_vllm_running():
        return False
    
    # 检查 HTTP 接口是否响应
    try:
        if TranslationState.http_client:
            resp = await TranslationState.http_client.get(f"{VLLM_BASE_URL}/v1/models", timeout=5)
            return resp.status_code == 200
    except:
        pass
    return False

# ==================== HTTP 路由 (FastAPI 装饰器风格) ====================
@app.get("/health")
async def health():
    """健康检查 - 同时检测 vLLM 和代理服务"""
    vllm_ok = await check_vllm_health()
    proxy_ok = TranslationState.openai_client is not None
    
    if not vllm_ok or not proxy_ok:
        # 返回 503 触发容器重启
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "vllm": "ok" if vllm_ok else "down",
                "proxy": "ok" if proxy_ok else "down"
            }
        )
    
    return {
        "status": "healthy",
        "vllm": "ok",
        "proxy": "ok"
    }

@app.get("/")
async def root():
    """服务信息"""
    return {
        "service": "Translation Proxy",
        "endpoint": "/turing/itrans/v3/func",
        "sentence_split": ENABLE_TEXT_SPLIT,
        "vllm_running": TranslationState.is_vllm_running()
    }

@app.get("/stats")
async def stats():
    """性能统计"""
    result = {
        "requests": TranslationState.total_requests, 
        "texts": TranslationState.total_texts, 
        "time_ms": round(TranslationState.total_time_ms, 2)
    }
    if TranslationState.total_requests > 0:
        result["avg_ms"] = round(TranslationState.total_time_ms / TranslationState.total_requests, 2)
    return result

@app.get("/metrics")
@app.get("/v1/metrics")
async def metrics():
    """透传 vLLM metrics"""
    try:
        resp = await TranslationState.http_client.get(f"{VLLM_BASE_URL}/metrics")
        return PlainTextResponse(resp.text)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/v1/models")
async def models():
    try:
        resp = await TranslationState.http_client.get(f"{VLLM_BASE_URL}/v1/models")
        return resp.json()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/turing/itrans/v3/func")
async def translate(request: Request):
    """
    请求: {"header": {"traceId": ""}, "parameter": {"engine": {"type": "cnen"}}, "payload": {"data": [...]}}
    响应: {"header": {"code": 0}, "payload": {"from": "cn", "to": "en", "result": [[{"src": "", "dst": ""}]]}}
    """
    start = time.time()
    try:
        body = await request.json()
        trace_id = body.get("header", {}).get("traceId", "")
        lang_type = body["parameter"]["engine"]["type"]
        texts = body["payload"]["data"]
        
        source_language_code, target_language_code = parse_lang_type(lang_type)
        source_language_name = LANGUAGE_MAP.get(source_language_code, source_language_code)
        target_language_name = LANGUAGE_MAP.get(target_language_code, target_language_code)
        
        translation_error: Optional[TranslationError] = None
        
        # 期望默认为切分（如果未设置，则切分）
        if ENABLE_TEXT_SPLIT:
            all_segs, seg_counts = [], []
            for t in texts:
                segs = split_sentences(t)
                all_segs.extend(segs)
                seg_counts.append(len(segs))
            logger.info(f"[{trace_id}] {source_language_name}->{target_language_name}, 原始:{len(texts)}条, 切分:{len(all_segs)}条")
            all_trans, translation_error = await translate_batch(all_segs, source_language_name, target_language_name)
            translations, idx = [], 0
            for cnt in seg_counts:
                translations.append("".join(all_trans[idx:idx+cnt]))
                idx += cnt
        else:
            logger.info(f"[{trace_id}] {source_language_name}->{target_language_name}, {len(texts)}条")
            translations, translation_error = await translate_batch(texts, source_language_name, target_language_name)
        
        elapsed = (time.time() - start) * 1000
        TranslationState.total_requests += 1
        TranslationState.total_texts += len(texts)
        TranslationState.total_time_ms += elapsed
        
        # 如果有翻译错误，返回对应的 HTTP 状态码
        if translation_error:
            failed_count = sum(1 for t in translations if t == "")
            logger.warning(f"[{trace_id}] 部分失败: {failed_count}/{len(texts)}条, 错误: {translation_error.message}")
            return JSONResponse(
                status_code=translation_error.status_code,
                content={
                    "header": {
                        "code": translation_error.status_code, 
                        "message": translation_error.message, 
                        "traceId": trace_id
                    },
                    "payload": {
                        "from": source_language_code, 
                        "to": target_language_code,
                        "result": [[{"src": s, "dst": d}] for s, d in zip(texts, translations)]
                    }
                }
            )
        
        logger.info(f"[{trace_id}] 完成, 耗时:{elapsed:.0f}ms, 平均:{elapsed/max(len(texts),1):.1f}ms/条")
        return {
            "header": {"code": 0, "message": "success", "traceId": trace_id},
            "payload": {"from": source_language_code, "to": target_language_code,
                       "result": [[{"src": s, "dst": d}] for s, d in zip(texts, translations)]}
        }
    except KeyError as e:
        return JSONResponse({"header": {"code": 400, "message": f"缺少字段: {e}"}, "payload": None}, status_code=400)
    except TranslationError as e:
        logger.error(f"翻译异常: {e.message}")
        return JSONResponse(
            status_code=e.status_code,
            content={"header": {"code": e.status_code, "message": e.message}, "payload": None}
        )
    except Exception as e:
        logger.error(f"异常: {e}")
        return JSONResponse({"header": {"code": 500, "message": str(e)}, "payload": None}, status_code=500)

# ==================== 启动入口 ====================
if __name__ == "__main__":
    uvicorn.run(
        "translation_proxy:app",
        host="0.0.0.0",
        port=PROXY_PORT,
        log_level="info"
    )

