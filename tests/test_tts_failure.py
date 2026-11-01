from __future__ import annotations

from typing import Any

from echobridge.llm import LLMDelta
from echobridge.protocol import ToolCall
from echobridge.tts import TtsUnavailableError
from tests.test_session import (
    index_of,
    speak_and_endpoint,
    start_session,
    stop_session,
)

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "current weather",
    "parameters": {"type": "object", "properties": {}},
}


class FailingTTS:
    """Yield one PCM chunk, then fail exactly like a dead vLLM worker."""

    def __init__(self, failures: int = 1) -> None:
        self.failures = failures
        self.calls = 0
        self.last_synthesis_at = 0.0

    async def synthesize(self, text: str, voice_id: str, generation: int):
        self.calls += 1
        # The first chunk reaches the pacer; the next iteration is the
        # worker's 500/engine-dead error.
        yield b"\x00\x00" * 4_800
        if self.calls <= self.failures:
            raise TtsUnavailableError("mock TTS worker is down")

    def cached(self, text: str, voice_id: str) -> bytes | None:
        return None

    async def warmup(self) -> None:
        return None

    async def ensure_warm(self, voice_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


class ScriptedLLM:
    """Call get_weather once, then answer from the tool result in history."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    async def warm(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def stream(self, messages, tools):
        self.calls.append(messages)
        has_weather = any(
            item.get("role") == "tool" and item.get("name") == "get_weather"
            for item in messages
        )
        if has_weather:
            yield LLMDelta(text="今天晴，二十度。")
            yield LLMDelta(done=True)
            return
        yield LLMDelta(text="我查一下哦。")
        yield LLMDelta(
            tool_calls=[ToolCall("call_weather", "get_weather", "{}")],
            done=True,
        )


def _failed_dones(ws) -> list[dict[str, Any]]:
    return [
        event
        for event in ws.events("response.done")
        if event.get("status") == "failed"
    ]


async def test_tts_failure_keeps_text_reason_and_tool_history():
    tts = FailingTTS(failures=1)
    llm = ScriptedLLM()
    ws, session, task = await start_session(
        mock_asr_text="今天天气怎么样",
        tools=[WEATHER_TOOL],
        tts=tts,
        llm=llm,
    )
    try:
        await speak_and_endpoint(ws)

        tool = await ws.wait_for_type("response.function_call_arguments.done")
        assert [item["name"] for item in tool["items"]] == ["get_weather"]

        # The client executes the tool even though audio failed.
        await ws.push(
            {
                "type": "conversation.item.create",
                "items": [
                    {
                        "type": "message",
                        "role": "tool",
                        "call_id": tool["items"][0]["call_id"],
                        "content": [{"type": "input_text", "text": "晴，20 度"}],
                    }
                ],
            }
        )

        await ws.wait_until(lambda: bool(_failed_dones(ws)))
        failed = _failed_dones(ws)[0]
        text_done = ws.events("response.output_text.done")[0]

        # (a) the generated text is on the wire even though synthesis failed.
        assert text_done["text"] == "我查一下哦。"
        assert index_of(ws, "response.output_text.done") < index_of(
            ws, "response.done"
        )
        # (b) real failure, with a machine-readable reason.  `failed` is not
        # remapped to `stopped`.
        assert failed["status"] == "failed"
        assert failed["reason"] == "tts_unavailable"

        # (c) the failed turn is in history with its tool call and result.
        assistant = [
            item
            for item in session._history
            if item.get("role") == "assistant"
            and item.get("tool_calls")
        ]
        tool_messages = [
            item for item in session._history if item.get("role") == "tool"
        ]
        assert len(assistant) == 1
        assert assistant[0]["content"] == "我查一下哦。"
        assert assistant[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert [item["name"] for item in tool_messages] == ["get_weather"]
        assert tool_messages[0]["content"] == "晴，20 度"

        # (d) the next turn must not repeat get_weather.  The scripted model
        # answers only when the tool result is present in the request.
        await speak_and_endpoint(ws)
        await ws.wait_until(lambda: len(ws.events("response.done")) >= 2)
        second = ws.events("response.done")[-1]
        assert second["status"] == "completed"
        assert "reason" not in second
        assert len(ws.events("response.function_call_arguments.done")) == 1
        assert any(
            item.get("role") == "tool" and item.get("name") == "get_weather"
            for item in llm.calls[-1]
        )
        assert ws.events("response.output_text.done")[-1]["text"] == "今天晴，二十度。"
    finally:
        await stop_session(ws, session, task)
