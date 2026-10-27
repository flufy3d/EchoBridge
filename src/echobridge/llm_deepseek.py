"""DeepSeek Flash streaming adapter (OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import Settings
from .llm import LLMDelta
from .logging import get_logger
from .protocol import ToolCall

logger = get_logger("echobridge.deepseek")

REALTIME_VOICE_PROMPT = """【实时语音约束（优先于上文字数限制）】
1. 每次回复 1–2 句，口语化，不超过 60 个汉字。
2. 不要 Markdown、列表、标题、代码块、括号动作、表情符号、舞台说明。
3. 不要提“专家/工具/照片/API/模型”；不要解释推理过程。
4. 需要看用户眼前画面时调用 look_and_ask；不要猜。
5. 不要调用 ask_expert（该工具在本后端不可用）；复杂问题直接简短回答或说明。
6. 句末要顺口，避免长串数字、英文缩写和生僻符号。"""


def build_system_prompt(instructions: str) -> str:
    base = (instructions or "").strip()
    return f"{base}\n\n{REALTIME_VOICE_PROMPT}" if base else REALTIME_VOICE_PROMPT


class DeepSeekLLM:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise ValueError("ECHOBRIDGE_LLM_API_KEY is required for llm_backend=deepseek")
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.llm_total_timeout_s,
                connect=5.0,
                read=settings.llm_total_timeout_s,
            )
        )

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMDelta]:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": True,
            "max_tokens": self._settings.llm_max_tokens,
            "temperature": self._settings.llm_temperature,
            "reasoning_effort": "low",
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            # Cloudflare on opencode.ai rejects some library UAs.
            "User-Agent": "curl/8.0",
        }

        started = time.perf_counter()
        for attempt in range(3):
            try:
                async with self._client.stream(
                    "POST", self._settings.llm_url, json=payload, headers=headers
                ) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        body = (await response.aread()).decode("utf-8", "replace")
                        if attempt < 2:
                            delay = 1.0 if attempt == 0 else 2.0
                            logger.warning(
                                "llm http %d, retry in %.1fs", response.status_code, delay
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise RuntimeError(
                            f"DeepSeek HTTP {response.status_code}: {body[:200]}"
                        )
                    if response.status_code != 200:
                        body = (await response.aread()).decode("utf-8", "replace")
                        raise RuntimeError(
                            f"DeepSeek HTTP {response.status_code}: {body[:200]}"
                        )
                    async for delta in self._parse_sse(response):
                        yield delta
                    logger.info("llm stream done in %.0f ms", (time.perf_counter() - started) * 1000)
                    return
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt < 2:
                    await asyncio.sleep(1.0 if attempt == 0 else 2.0)
                    continue
                raise
        raise RuntimeError("DeepSeek request failed after retries")

    async def _parse_sse(self, response: httpx.Response) -> AsyncIterator[LLMDelta]:
        lines = response.aiter_lines().__aiter__()
        started = time.monotonic()
        first_content = False
        first_tool = False
        accumulated: dict[int, dict[str, str]] = {}
        finish_reason: str | None = None
        first_content_deadline = started + self._settings.llm_first_token_timeout_s
        total_deadline = started + self._settings.llm_total_timeout_s

        while True:
            now = time.monotonic()
            if now >= total_deadline:
                raise TimeoutError("DeepSeek total timeout")
            timeout = total_deadline - now
            if not first_content and not first_tool:
                timeout = min(timeout, max(0.1, first_content_deadline - now))
            try:
                line = await asyncio.wait_for(anext(lines), timeout=timeout)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError as exc:
                if not first_content and not first_tool:
                    raise TimeoutError("DeepSeek first content timeout") from exc
                raise TimeoutError("DeepSeek stream timeout") from exc
            if not line:
                continue
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                logger.debug("bad SSE JSON ignored")
                continue
            if event.get("error"):
                raise RuntimeError(f"DeepSeek error: {event['error']}")
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            # reasoning_content is deliberately never forwarded.
            content = delta.get("content")
            if isinstance(content, str) and content:
                if not first_content:
                    first_content = True
                    logger.info(
                        "llm first content %.0f ms", (time.monotonic() - started) * 1000
                    )
                yield LLMDelta(text=content)
            for tool_delta in delta.get("tool_calls") or []:
                if not isinstance(tool_delta, dict):
                    continue
                index = int(tool_delta.get("index") or 0)
                slot = accumulated.setdefault(
                    index, {"id": "", "name": "", "arguments": ""}
                )
                if tool_delta.get("id"):
                    slot["id"] = str(tool_delta["id"])
                function = tool_delta.get("function") or {}
                if function.get("name"):
                    slot["name"] = str(function["name"])
                if function.get("arguments"):
                    slot["arguments"] += str(function["arguments"])
                first_tool = True
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

        calls: list[ToolCall] = []
        for index in sorted(accumulated):
            slot = accumulated[index]
            if not slot.get("name"):
                continue
            calls.append(
                ToolCall(
                    call_id=slot.get("id") or f"call_{index}",
                    name=slot["name"],
                    arguments=slot.get("arguments") or "{}",
                )
            )
        if calls:
            logger.info("llm tool_calls=%d finish=%s", len(calls), finish_reason or "-")
            yield LLMDelta(tool_calls=calls, done=True)
        else:
            yield LLMDelta(done=True)

    async def close(self) -> None:
        await self._client.aclose()
