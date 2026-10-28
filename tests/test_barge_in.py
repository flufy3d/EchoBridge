from __future__ import annotations

import asyncio

import pytest

from echobridge.asr import ASRResult
from echobridge.config import Settings
from tests.helpers import loud_frame, make_settings, silence_frame
from tests.test_session import send_audio, start_session, stop_session

GREETING = "你来啦，我在呢。今天也要开开心心的呀，想聊点什么都可以哦。"


class SequenceASR:
    """Record the PCM sent to ASR and return one transcript per call."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls: list[bytes] = []

    async def transcribe(
        self, pcm: bytes, sample_rate: int = 16000, context: str | None = None
    ) -> ASRResult:
        self.calls.append(pcm)
        index = min(len(self.calls) - 1, len(self._texts) - 1)
        return ASRResult(self._texts[index], "zh", 0.0)

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


def duplex_settings(**overrides) -> Settings:
    base = dict(
        backchannel_enabled=False,
        tts_first_chunk_ms=1,
        # Long enough that the greeting response is still active while the
        # test injects barge-in audio.
        tts_tone_char_ms=50,
        vad_endpoint_ms=400,
        vad_min_speech_ms=200,
    )
    base.update(overrides)
    return make_settings(**base)


async def start_greeting(ws) -> str:
    await ws.push(
        {
            "type": "speech_text_buffer.append",
            "speech_id": "g1",
            "text": GREETING,
        }
    )
    await ws.push(
        {
            "type": "speech_text_buffer.commit",
            "speech_id": "g1",
            "text": "",
        }
    )
    started = await ws.wait_for_type("response.output_audio.started")
    return str(started["response_id"])


def user_turns(session) -> list[dict]:
    return [item for item in session._history if item.get("role") == "user"]


@pytest.mark.parametrize(
    "cancel_event",
    [
        {"type": "response.cancel", "reason": "barge_in"},
        # Older apps omit reason; an active VAD segment is still preserved.
        {"type": "response.cancel"},
    ],
    ids=["reason_barge_in", "legacy_no_reason"],
)
async def test_full_duplex_keeps_first_frames_and_starts_one_turn(cancel_event):
    asr = SequenceASR(["给我讲个故事"])
    ws, session, task = await start_session(
        settings=duplex_settings(),
        full_duplex=True,
        aec="hw-xmvoip",
        asr=asr,
    )
    try:
        old_id = await start_greeting(ws)
        assert session._full_duplex is True
        assert session._input_aec == "hw-xmvoip"

        # The unique high-amplitude frames represent the first word the old
        # half-duplex path used to lose.  Keep them distinct from the rest so
        # the ASR payload can prove they were preserved across response.cancel.
        first_word = [loud_frame(amplitude=12000) for _ in range(3)]
        rest = [loud_frame(amplitude=9000) for _ in range(12)]
        await send_audio(ws, first_word)
        await ws.push(cancel_event)
        await send_audio(ws, rest + [silence_frame() for _ in range(20)])

        completed = await ws.wait_for_type(
            "conversation.item.input_audio_transcription.completed"
        )
        assert completed["transcript"] == "给我讲个故事"
        assert asr.calls
        assert b"".join(first_word) in asr.calls[0]
        # No asr.started while TTS was active: the app must not flush its
        # local player until the response itself is canceled.
        assert not ws.events(
            "conversation.item.input_audio_transcription.started"
        )

        canceled = ws.events("response.canceled")
        assert [item["response_id"] for item in canceled] == [old_id]

        done = await ws.wait_for_type("response.done")
        assert done["response_id"] != old_id
        assert [item["content"] for item in user_turns(session)] == [
            "给我讲个故事"
        ]
    finally:
        await stop_session(ws, session, task)


async def test_barge_in_transcript_keeps_first_character():
    asr = SequenceASR(["等一下，先别讲了"])
    ws, session, task = await start_session(
        settings=duplex_settings(),
        full_duplex=True,
        aec="hw-xmvoip",
        asr=asr,
    )
    try:
        old_id = await start_greeting(ws)
        first_word = [loud_frame(amplitude=12000) for _ in range(3)]
        rest = [loud_frame(amplitude=9000) for _ in range(12)]
        await send_audio(ws, first_word)
        await ws.push({"type": "response.cancel", "reason": "barge_in"})
        await send_audio(ws, rest + [silence_frame() for _ in range(20)])

        completed = await ws.wait_for_type(
            "conversation.item.input_audio_transcription.completed"
        )
        assert completed["transcript"].startswith("等")
        assert b"".join(first_word) in asr.calls[0]

        done = await ws.wait_for_type("response.done")
        assert done["status"] == "stopped"
        assert done["response_id"] != old_id
        assert [item["response_id"] for item in ws.events("response.canceled")] == [
            old_id
        ]
        assert not user_turns(session)
        assert not ws.events("response.output_text.delta")
    finally:
        await stop_session(ws, session, task)


async def test_stop_command_during_response_does_not_start_a_turn():
    asr = SequenceASR(["停一下"])
    ws, session, task = await start_session(
        settings=duplex_settings(),
        full_duplex=True,
        aec="hw-xmvoip",
        asr=asr,
    )
    try:
        old_id = await start_greeting(ws)
        await send_audio(
            ws,
            [loud_frame() for _ in range(12)]
            + [silence_frame() for _ in range(20)],
        )
        canceled = await ws.wait_for_type("response.canceled")
        assert canceled["response_id"] == old_id
        completed = await ws.wait_for_type(
            "conversation.item.input_audio_transcription.completed"
        )
        assert completed["transcript"] == "停一下"

        await asyncio.sleep(0.1)
        assert not session._history
        assert not ws.events("response.output_text.delta")
        assert not ws.events("response.done")
        assert {
            item["response_id"]
            for item in ws.events("response.output_audio.started")
        } == {old_id}
        assert len(ws.events("response.canceled")) == 1
    finally:
        await stop_session(ws, session, task)


async def test_short_noise_during_response_does_not_cancel_tts():
    asr = SequenceASR(["给我讲个故事"])
    ws, session, task = await start_session(
        settings=duplex_settings(),
        full_duplex=True,
        aec="hw-xmvoip",
        asr=asr,
    )
    try:
        old_id = await start_greeting(ws)
        # 60 ms of noise is below vad_min_speech_ms.  It may start and end a
        # VAD segment, but must not interrupt the response.
        await send_audio(
            ws,
            [loud_frame() for _ in range(3)]
            + [silence_frame() for _ in range(20)],
        )
        failed = await ws.wait_for_type(
            "conversation.item.input_audio_transcription.failed"
        )
        assert failed["error"] == "empty speech"
        await asyncio.sleep(0.1)
        assert session._response_active is True
        assert session._active_ctx is not None
        assert session._active_ctx.response_id == old_id
        assert not asr.calls
        assert not ws.events("response.canceled")
        assert not ws.events("response.done")
    finally:
        await stop_session(ws, session, task)


async def test_tap_cancel_resets_vad_and_does_not_start_a_turn():
    asr = SequenceASR(["给我讲个故事"])
    ws, session, task = await start_session(
        settings=duplex_settings(),
        full_duplex=True,
        aec="hw-xmvoip",
        asr=asr,
    )
    try:
        old_id = await start_greeting(ws)
        await send_audio(ws, [loud_frame() for _ in range(12)])
        await ws.push(
            {
                "type": "response.cancel",
                "reason": "tap",
                "response_id": old_id,
            }
        )
        canceled = await ws.wait_for_type("response.canceled")
        assert canceled["response_id"] == old_id

        # If tap had left the segment active, this silence would endpoint it
        # and open a new turn.
        await send_audio(ws, [silence_frame() for _ in range(20)])
        await asyncio.sleep(0.1)
        assert session._vad.active is False
        assert not asr.calls
        assert not user_turns(session)
        assert not ws.events(
            "conversation.item.input_audio_transcription.completed"
        )
        assert not ws.events("response.done")
    finally:
        await stop_session(ws, session, task)


async def test_half_duplex_ignores_audio_during_response():
    asr = SequenceASR(["给我讲个故事"])
    ws, session, task = await start_session(
        settings=duplex_settings(),
        full_duplex=False,
        asr=asr,
    )
    try:
        old_id = await start_greeting(ws)
        await send_audio(
            ws,
            [loud_frame() for _ in range(12)]
            + [silence_frame() for _ in range(20)],
        )
        await asyncio.sleep(0.1)

        assert session._full_duplex is False
        assert session._vad.active is False
        assert not asr.calls
        assert not user_turns(session)
        assert not ws.events("response.canceled")
        assert not ws.events(
            "conversation.item.input_audio_transcription.completed"
        )
        assert not ws.events("response.done")
        assert {
            item["response_id"]
            for item in ws.events("response.output_audio.started")
        } == {old_id}
    finally:
        await stop_session(ws, session, task)


async def test_duplicate_cancel_sends_canceled_once():
    ws, session, task = await start_session(
        settings=duplex_settings(), full_duplex=True, aec="hw-xmvoip"
    )
    try:
        old_id = await start_greeting(ws)
        await ws.push({"type": "response.cancel", "reason": "client"})
        await ws.push({"type": "response.cancel", "reason": "client"})
        canceled = await ws.wait_for_type("response.canceled")
        assert canceled["response_id"] == old_id
        await asyncio.sleep(0.1)
        assert len(ws.events("response.canceled")) == 1
    finally:
        await stop_session(ws, session, task)


async def test_stale_response_cancel_does_not_cancel_new_response():
    ws, session, task = await start_session(
        settings=duplex_settings(), full_duplex=True, aec="hw-xmvoip"
    )
    try:
        old_id = await start_greeting(ws)
        await ws.push({"type": "response.cancel", "reason": "client"})
        await ws.wait_for_type("response.canceled")

        ws.sent.clear()
        await start_greeting(ws)
        started = ws.events("response.output_audio.started")[-1]
        new_id = started["response_id"]
        assert new_id != old_id

        # A delayed cancel that names the previous response must not touch
        # the new turn.
        await ws.push(
            {
                "type": "response.cancel",
                "reason": "barge_in",
                "response_id": old_id,
            }
        )
        await asyncio.sleep(0.1)
        assert not ws.events("response.canceled")
        assert session._response_active is True
        assert session._active_ctx is not None
        assert session._active_ctx.response_id == new_id
    finally:
        await stop_session(ws, session, task)


async def test_missing_extension_defaults_to_half_duplex():
    ws, session, task = await start_session(settings=duplex_settings())
    try:
        assert session._full_duplex is False
        assert session._input_aec == ""
    finally:
        await stop_session(ws, session, task)


async def test_env_full_duplex_override_without_extension():
    ws, session, task = await start_session(
        settings=duplex_settings(full_duplex=True)
    )
    try:
        assert session._full_duplex is True
        assert session._input_aec == ""
    finally:
        await stop_session(ws, session, task)
