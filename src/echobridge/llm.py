"""Phase 1 mock LLM.

The real DeepSeek adapter lands in Phase 4.  This module keeps the session
logic honest: text streams character by character, and a turn can produce one or
more function calls before the final spoken text.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

from .protocol import ToolCall


@dataclass(slots=True)
class MockPlan:
    text_before: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    text_after: str = ""


@dataclass(slots=True)
class LLMDelta:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    done: bool = False


class MockLLM:
    def __init__(self, char_delay_ms: int = 12) -> None:
        self._char_delay = max(0, char_delay_ms) / 1000.0

    def plan(self, transcript: str, available_tools: set[str]) -> MockPlan:
        text = transcript.strip()
        lower = text.lower()

        def call(name: str, arguments: dict) -> ToolCall:
            return ToolCall(
                call_id=f"call_{uuid.uuid4().hex[:10]}",
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            )

        # Explicit batch test: two calls in one items array.
        if ("表情" in text or "笑" in text) and (
            "动作" in text or "转圈" in text or "转个圈" in text
        ):
            calls = []
            if "set_emotion" in available_tools:
                calls.append(call("set_emotion", {"emotion": "happy"}))
            if "play_animation" in available_tools:
                calls.append(call("play_animation", {"name": "spin"}))
            if calls:
                return MockPlan(tool_calls=calls, text_after="好呀，看我笑一个，再转个圈。")

        if ("表情" in text or "笑" in text) and "set_emotion" in available_tools:
            return MockPlan(
                tool_calls=[call("set_emotion", {"emotion": "happy"})],
                text_after="好呀，看我笑一个。",
            )

        if ("动作" in text or "转圈" in text or "转个圈" in text) and "play_animation" in available_tools:
            return MockPlan(
                tool_calls=[call("play_animation", {"name": "spin"})],
                text_after="好呀，转个圈。",
            )

        if ("时间" in text or "几点" in text) and "get_time" in available_tools:
            return MockPlan(
                tool_calls=[call("get_time", {})],
                text_after="我看看现在几点啦。",
            )

        if ("钢盆" in text or "道具" in text) and "set_prop" in available_tools:
            return MockPlan(
                tool_calls=[call("set_prop", {"name": "steel_basin", "enabled": True})],
                text_after="戴上啦，好看吗？",
            )

        if ("看看" in text or "这是什么" in text) and "look_and_ask" in available_tools:
            return MockPlan(
                tool_calls=[call("look_and_ask", {"question": text, "context": ""})],
                text_after="我看看哦。",
            )

        if "再见" in text and "end_conversation" in available_tools:
            return MockPlan(
                tool_calls=[call("end_conversation", {})],
                text_after="拜拜，下次见。",
            )

        if "你好" in text or "在吗" in text or lower in {"hi", "hello"}:
            return MockPlan(text_after="你好呀，我在呢。")
        if "今天" in text:
            return MockPlan(text_after="今天也要开开心心的呀。")
        return MockPlan(text_after="我在呢，今天想聊点什么呀？")

    async def stream_text(self, text: str) -> AsyncIterator[str]:
        for char in text:
            if self._char_delay:
                await asyncio.sleep(self._char_delay)
            yield char
