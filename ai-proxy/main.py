"""
AI Thinking Proxy
=================
透明代理服务，接收 OpenAI 兼容的 /v1/chat/completions 请求，
根据 model 字段自动匹配后端配置，注入 thinking-disable 参数，
转发到真实 AI 后端，处理双保险重试逻辑。

支持流式（SSE）和非流式两种模式透传。
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import fnmatch
import httpx
import yaml
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

# ── 加载配置 ──────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    PROXY_CFG = yaml.safe_load(f) or {}

# ── 日志 ──────────────────────────────────────────────────────────────────

log_level = getattr(logging, PROXY_CFG.get("log_level", "INFO").upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s [AI-PROXY] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ai-proxy")

# ── FastAPI ────────────────────────────────────────────────────────────────

_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=30.0)
http_client: Optional[httpx.AsyncClient] = None


def _create_app() -> FastAPI:
    """创建 FastAPI 应用（工厂函数，方便测试时重置状态）。"""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global http_client
        http_client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        logger.info("HTTP client initialized")
        yield
        if http_client:
            await http_client.aclose()
            http_client = None
            logger.info("HTTP client closed")

    app = FastAPI(title="AI Thinking Proxy", lifespan=lifespan)
    return app


app = _create_app()

# ── 辅助函数 ──────────────────────────────────────────────────────────────


def _match_backend(model_name: str) -> Optional[dict]:
    """使用 glob 通配符匹配 model 对应的后端配置。"""
    for backend in PROXY_CFG.get("backends", []):
        for pattern in backend.get("models", []):
            if fnmatch.fnmatch(model_name, pattern):
                logger.debug("Backend '%s' matched model '%s' (pattern: %s)", backend["name"], model_name, pattern)
                return backend
    return None


def _deep_inject(target: dict, source: dict):
    """递归地将 source 注入 target，支持嵌套 dict 的合并。"""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_inject(target[key], value)
        else:
            target[key] = value


async def _forward_sync(
    http_client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    thinking_param: dict,
    retry_on_400: bool,
) -> Response:
    """非流式转发，包含双保险重试逻辑。"""
    # 第一次请求（携带 thinking 禁用参数）
    logger.info("→ [%s] POST %s (stream=false, thinking_disabled=%s)", 
                payload.get("model", "?"), url, "yes" if thinking_param else "no")
    t0 = time.monotonic()
    resp = await http_client.post(url, headers=headers, json=payload)
    elapsed = time.monotonic() - t0

    # 双保险：400 错误时去掉 thinking 参数重试
    if resp.status_code == 400 and retry_on_400 and thinking_param:
        clean_payload = {k: v for k, v in payload.items() if k not in thinking_param}
        logger.warning("← 400, retrying without thinking-disabled params → [%s]", payload.get("model", "?"))
        t0 = time.monotonic()
        resp = await http_client.post(url, headers=headers, json=clean_payload)
        elapsed = time.monotonic() - t0

    logger.info("← %s (%.2fs)", resp.status_code, elapsed)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
    )


async def _forward_stream(
    http_client: httpx.AsyncClient,
    url: str,
    headers: dict,
    payload: dict,
    thinking_param: dict,
    retry_on_400: bool,
):
    """流式转发（SSE），包含双保险重试逻辑。"""
    # 先尝试带 thinking 禁用参数的请求
    logger.info("→ [%s] POST %s (stream=true, thinking_disabled=%s)", 
                payload.get("model", "?"), url, "yes" if thinking_param else "no")
    t0 = time.monotonic()
    req = http_client.build_request("POST", url, headers=headers, json=payload)
    resp = await http_client.send(req, stream=True)

    # 双保险：400 时重试
    if resp.status_code == 400 and retry_on_400 and thinking_param:
        await resp.aclose()
        clean_payload = {k: v for k, v in payload.items() if k not in thinking_param}
        logger.warning("← 400, retrying without thinking-disabled params (stream) → [%s]", payload.get("model", "?"))
        t0 = time.monotonic()
        req = http_client.build_request("POST", url, headers=headers, json=clean_payload)
        resp = await http_client.send(req, stream=True)

    logger.info("← %s (stream, %.2fs)", resp.status_code, time.monotonic() - t0)

    async def event_stream():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        event_stream(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/event-stream"),
    )


# ── 路由 ──────────────────────────────────────────────────────────────────


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    """代理 /v1/chat/completions 端点。"""
    body = await request.json()
    model = body.get("model", "")
    is_stream = body.get("stream", False)

    # 1. 匹配后端
    backend = _match_backend(model)
    if not backend:
        logger.warning("No backend configured for model '%s', returning 400", model)
        return Response(
            content=json.dumps({
                "error": f"No backend configured for model '{model}'. "
                         f"Add a matching entry in ai-proxy/config.yaml backends[].models."
            }),
            status_code=400,
            media_type="application/json",
        )

    # 2. 注入 thinking 禁用参数
    thinking_param = backend.get("thinking_param", {})
    if thinking_param:
        logger.info("注入 thinking-disabled 参数: %s → model=%s", thinking_param, model)
        _deep_inject(body, thinking_param)

    # 3. 构建转发目标
    base_url = backend["base_url"].rstrip("/")
    target_url = f"{base_url}/chat/completions"

    # 4. 构建请求头（透传 Authorization）
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth

    retry_on_400 = PROXY_CFG.get("retry_on_400", True)

    # 5. 选择转发模式（使用应用级 httpx client，确保流式连接不中断）
    client = http_client
    if client is None:
        return Response(
            content=json.dumps({"error": "Proxy not ready (HTTP client not initialized)"}),
            status_code=503,
            media_type="application/json",
        )
    if is_stream:
        return await _forward_stream(
            client, target_url, headers, body,
            thinking_param, retry_on_400,
        )
    else:
        return await _forward_sync(
            client, target_url, headers, body,
            thinking_param, retry_on_400,
        )


@app.get("/health")
async def health():
    """健康检查端点。"""
    return {
        "status": "ok",
        "backends": [b["name"] for b in PROXY_CFG.get("backends", [])],
    }


# ── 入口 ──────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    host = PROXY_CFG.get("listen_host", "127.0.0.1")
    port = PROXY_CFG.get("listen_port", 9997)
    logger.info("Starting AI Proxy on %s:%s", host, port)
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
