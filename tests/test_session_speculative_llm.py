from __future__ import annotations

import asyncio
import base64

from echobridge.asr import MockASR
from echobridge.llm import LLMDelta
from echobridge.session import Session
from echobridge.vad import VADEvent
from tests.fakes import FakeWebSocket
from tests.helpers import (
    loud_frame,
    make_settings,
    make_voice_registry,
    session_create,
)


class ScriptedVAD:
    """Deterministic early/final events; the energy VAD has no early event."""

    def __init__(self) -> None:
        self.frames = 0
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self.frames = 0
        self._active = False

    def set_playback(self, playing: bool) -> None:
        return None

    def push(self, pcm: bytes):
        self.frames += 1
        if self.frames == 1:
            self._active = True
            return [VADEvent("started", "item_1")]
        if self.frames == 5:
            return [
                VADEvent("early", "item_1", speech_ms=400, pcm=loud_frame() * 3)
            ]
        if self.frames == 8:
            self._active = False
            return [
                VADEvent("final", "item_1", speech_ms=600, pcm=loud_frame() * 10)
            ]
        return []


class FakeDeepSeek:
    def __init__(self) -> None:
        self.calls: list[list[dict]] = []

    async def stream(self, messages, tools):
        self.calls.append(messages)
        yield LLMDelta(text="你好呀。")
        yield LLMDelta(done=True)

    async def warm(self) -> None:
        return None

    async def close(self) -> None:
        return None


async def _push_audio(ws: FakeWebSocket, frames: list[bytes]) -> None:
    for frame in frames:
        await ws.push(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode("ascii"),
            }
        )


async def test_final_turn_reuses_the_speculative_llm_request(tmp_path):
    ws = FakeWebSocket()
    llm = FakeDeepSeek()
    session = Session(
        ws,
        make_settings(),
        device_id="test-device",
        voice_registry=make_voice_registry(tmp_path),
        asr=MockASR("你好呀，溟月"),
        vad=ScriptedVAD(),
        llm=llm,
    )
    task = asyncio.create_task(session.run())
    try:
        await ws.push(session_create([]))
        await ws.wait_for_type("session.created")

        # Early fires on frame 5; the LLM request starts before frame 8 emits
        # the final ASR result.
        await _push_audio(ws, [loud_frame() for _ in range(5)])
        await ws.wait_until(lambda: session._spec_llm is not None, timeout=2.0)
        assert len(llm.calls) == 1

        await _push_audio(ws, [loud_frame() for _ in range(3)])
        await ws.wait_for_type("response.output_audio.delta")
        await ws.wait_for_type("response.done")

        assert len(llm.calls) == 1
        assert llm.calls[0][-1] == {"role": "user", "content": "你好呀，溟月"}
        assert any(
            event.get("type") == "response.output_text.delta"
            and event.get("delta") == "你好呀。"
            for event in ws.sent
        )
    finally:
        if not session.closed:
            await session.close("test")
        try:
            await asyncio.wait_for(task, timeout=2)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
