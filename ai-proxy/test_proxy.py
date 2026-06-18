"""
AI Proxy 测试
=============
测试覆盖：模型匹配、参数注入、转发逻辑、双保险重试。
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from main import app, _match_backend, _deep_inject

client = TestClient(app)


# ── 测试夹具：提供 mock HTTP client ──────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_http_client():
    """每个测试前注入 mock HTTP client，确保转发不依赖真实连接。"""
    mock = MagicMock(spec=httpx.AsyncClient)
    mock.post = AsyncMock()
    mock.send = AsyncMock()
    mock.build_request = MagicMock()
    main.http_client = mock
    yield mock
    main.http_client = None


# ── 单元测试：模型匹配 ──────────────────────────────────────────────────


class TestMatchBackend:
    """测试 glob 通配符模型匹配逻辑。"""

    def test_match_llamacpp_model(self):
        """Qwen GGUF 模型匹配 llama-cpp 后端。"""
        backend = _match_backend("Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q4_K_M.gguf")
        assert backend is not None
        assert backend["name"] == "llama-cpp"

    def test_match_qwen_model(self):
        """qwen* 通配符匹配。"""
        backend = _match_backend("qwen2.5-7b")
        assert backend is not None
        assert backend["name"] == "llama-cpp"

    def test_match_deepseek_model(self):
        """DeepSeek 模型匹配。"""
        backend = _match_backend("deepseek-chat")
        assert backend is not None
        assert backend["name"] == "deepseek-api"

    def test_match_gemini_model(self):
        """Gemini 模型匹配。"""
        backend = _match_backend("gemini-2.5-pro")
        assert backend is not None
        assert backend["name"] == "gemini-api"

    def test_match_openai_o_model(self):
        """OpenAI o 系列匹配。"""
        backend = _match_backend("o3-mini")
        assert backend is not None
        assert backend["name"] == "openai-o-series"
        backend = _match_backend("o1-preview")
        assert backend is not None
        assert backend["name"] == "openai-o-series"

    def test_catch_all_vllm(self):
        """未在其他后端匹配的模型应被 catch-all 通配符 * 捕获。"""
        backend = _match_backend("unknown-model-v42")
        assert backend is not None
        assert backend["name"] == "catch-all"

    def test_no_backend_empty_list(self):
        """无后端配置时应返回 None。"""
        with patch("main.PROXY_CFG", {"backends": []}):
            from main import _match_backend as mb
            assert mb("any-model") is None


# ── 单元测试：参数注入 ──────────────────────────────────────────────────


class TestDeepInject:
    """测试深度参数注入逻辑。"""

    def test_simple_value(self):
        """简单键值注入。"""
        target = {"a": 1}
        _deep_inject(target, {"b": 2})
        assert target == {"a": 1, "b": 2}

    def test_nested_dict_merge(self):
        """嵌套字典合并。"""
        target = {"existing": {"x": 1}}
        _deep_inject(target, {"existing": {"y": 2}})
        assert target == {"existing": {"x": 1, "y": 2}}

    def test_nested_dict_override(self):
        """嵌套字典中相同 key 覆写。"""
        target = {"chat_template_kwargs": {"enable_thinking": True, "foo": "bar"}}
        _deep_inject(target, {"chat_template_kwargs": {"enable_thinking": False}})
        assert target["chat_template_kwargs"]["enable_thinking"] is False
        assert target["chat_template_kwargs"]["foo"] == "bar"

    def test_inject_thinking_disabled(self):
        """注入 llama.cpp 的 thinking-disable 参数。"""
        payload = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
        _deep_inject(payload, {"chat_template_kwargs": {"enable_thinking": False}})
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    def test_inject_deepseek_style(self):
        """注入 DeepSeek 风格的 thinking disabled 参数。"""
        payload = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]}
        _deep_inject(payload, {"thinking": {"type": "disabled"}})
        assert payload["thinking"]["type"] == "disabled"

    def test_inject_gemini_style(self):
        """注入 Gemini 风格的 thinking_config。"""
        payload = {"model": "gemini-2.5-pro", "messages": [{"role": "user", "content": "hi"}]}
        _deep_inject(payload, {"thinking_config": {"thinking_budget": 0}})
        assert payload["thinking_config"]["thinking_budget"] == 0

    def test_inject_openai_style(self):
        """注入 OpenAI o 系列的 reasoning_effort。"""
        payload = {"model": "o3-mini", "messages": [{"role": "user", "content": "hi"}]}
        _deep_inject(payload, {"reasoning_effort": "low"})
        assert payload["reasoning_effort"] == "low"


# ── 集成测试：API 端点 ──────────────────────────────────────────────────


class TestHealthEndpoint:
    """测试健康检查端点。"""

    def test_health_returns_backends(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "backends" in data


class TestProxyEndpoint:
    """测试代理转发端点。"""

    def test_proxy_no_model_match(self):
        """无匹配模型应返回 400。"""
        with patch("main.PROXY_CFG", {"backends": [], "retry_on_400": True}):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "no-match", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert resp.status_code == 400
            assert "No backend configured" in resp.text

    @pytest.mark.asyncio
    async def test_proxy_sync_forward_success(self, mock_http_client):
        """非流式转发成功。"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "choices": [{"message": {"content": "Hello!"}}]
        }).encode()
        mock_response.headers = {"content-type": "application/json"}

        mock_http_client.post.return_value = mock_response

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3.6-27B-test.gguf",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Hello!"

        # 验证请求体包含了 thinking-disable 参数
        call_kwargs = mock_http_client.post.call_args[1]
        sent_json = call_kwargs["json"]
        assert "chat_template_kwargs" in sent_json
        assert sent_json["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_proxy_400_retry(self, mock_http_client):
        """400 错误时应去掉 thinking 参数重试。"""
        mock_400 = MagicMock(spec=httpx.Response)
        mock_400.status_code = 400
        mock_400.content = b'{"error":"bad request"}'
        mock_400.headers = {"content-type": "application/json"}

        mock_200 = MagicMock(spec=httpx.Response)
        mock_200.status_code = 200
        mock_200.content = json.dumps({
            "choices": [{"message": {"content": "Retried OK!"}}]
        }).encode()
        mock_200.headers = {"content-type": "application/json"}

        mock_http_client.post.side_effect = [mock_400, mock_200]

        with patch("main.logger"):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "Qwen3.6-27B-test.gguf",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Retried OK!"

        # 验证第一次请求带 thinking 参数，第二次不带
        first_call = mock_http_client.post.call_args_list[0]
        first_json = first_call[1]["json"]
        assert "chat_template_kwargs" in first_json

        second_call = mock_http_client.post.call_args_list[1]
        second_json = second_call[1]["json"]
        assert "chat_template_kwargs" not in second_json

    @pytest.mark.asyncio
    async def test_proxy_stream_success(self, mock_http_client):
        """流式转发成功且 SSE 透传。"""
        # 模拟流式响应数据
        sse_chunks = [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo!"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]

        class MockStreamResponse:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def aiter_bytes(self):
                for chunk in sse_chunks:
                    yield chunk

            async def aclose(self):
                pass

        mock_http_client.send.return_value = MockStreamResponse()

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen3.6-27B-test.gguf",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )

        assert resp.status_code == 200
        # 验证 SSE 数据被透传
        lines = resp.text.strip().split("\n\n")
        assert len(lines) == 3
        assert "Hel" in lines[0]

    @pytest.mark.asyncio
    async def test_proxy_deepseek_model_thinking_injected(self, mock_http_client):
        """DeepSeek 模型注入正确的 thinking disabled 参数。"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.content = json.dumps({
            "choices": [{"message": {"content": "OK"}}]
        }).encode()
        mock_response.headers = {"content-type": "application/json"}

        mock_http_client.post.return_value = mock_response

        resp = client.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert resp.status_code == 200
        call_kwargs = mock_http_client.post.call_args[1]
        sent_json = call_kwargs["json"]
        assert "thinking" in sent_json
        assert sent_json["thinking"]["type"] == "disabled"
