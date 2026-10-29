from __future__ import annotations

import json

import httpx

from echobridge.llm_deepseek import DeepSeekLLM
from tests.helpers import make_settings

SSE = (
    'data: {"choices":[{"delta":{"content":"你"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _settings(**overrides):
    values = {
        "llm_backend": "deepseek",
        "llm_api_key": "test-key",
        "llm_url": "https://example.test/v1/chat/completions",
        "llm_model": "deepseek-v4.1-flash",
    }
    values.update(overrides)
    return make_settings(**values)


async def _client(settings) -> tuple[DeepSeekLLM, list[tuple[str, dict]]]:
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        requests.append((request.url.path, body))
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(
            200, text=SSE, headers={"content-type": "text/event-stream"}
        )

    llm = DeepSeekLLM(settings)
    await llm._client.aclose()
    llm._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return llm, requests


async def test_thinking_is_disabled_by_default():
    llm, requests = await _client(_settings())
    try:
        deltas = [
            delta
            async for delta in llm.stream([{"role": "user", "content": "你好"}], [])
        ]
    finally:
        await llm.close()

    payload = requests[0][1]
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert "".join(delta.text for delta in deltas) == "你"


async def test_thinking_can_be_enabled_and_tools_are_forwarded():
    llm, requests = await _client(_settings(llm_thinking="enabled"))
    tools = [{"type": "function", "function": {"name": "get_time"}}]
    try:
        async for _ in llm.stream([{"role": "user", "content": "你好"}], tools):
            pass
    finally:
        await llm.close()

    payload = requests[0][1]
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "low"
    assert payload["tools"] == tools
    assert payload["tool_choice"] == "auto"


async def test_warm_opens_the_pool_once():
    llm, requests = await _client(_settings())
    try:
        await llm.warm()
        await llm.warm()
    finally:
        await llm.close()

    assert [path for path, _ in requests] == ["/v1/models"]
