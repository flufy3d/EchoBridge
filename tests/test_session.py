from __future__ import annotations

import asyncio
import base64

from echobridge.config import Settings
from echobridge.protocol import dumps
from echobridge.session import Session
from echobridge.voices import VoiceRegistry
from tests.fakes import FakeWebSocket
from tests.helpers import (
    ask_expert_tool,
    get_time_tool,
    loud_frame,
    make_settings,
    make_voice_registry,
    play_animation_tool,
    session_create,
    set_emotion_tool,
    silence_frame,
)


async def start_session(
    *,
    mock_asr_text: str = "你好呀，溟月",
    tools: list[dict] | None = None,
    settings: Settings | None = None,
    voice_registry: VoiceRegistry | None = None,
    voice: str | None = "mingyue_v1",
    include_voice: bool = True,
    full_duplex: bool | None = None,
    aec: str | None = None,
    asr=None,
) -> tuple[FakeWebSocket, Session, asyncio.Task[None]]:
    ws = FakeWebSocket()
    session = Session(
        ws,
        settings or make_settings(),
        device_id="test-device",
        mock_asr_text=mock_asr_text,
        voice_registry=voice_registry,
        asr=asr,
    )
    task = asyncio.create_task(session.run())
    await ws.push(
        session_create(
            tools or [],
            voice=voice,
            include_voice=include_voice,
            full_duplex=full_duplex,
            aec=aec,
        )
    )
    await ws.wait_for_type("session.created")
    return ws, session, task


async def stop_session(ws: FakeWebSocket, session: Session, task: asyncio.Task[None]) -> None:
    if not session.closed:
        await session.close("test")
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def send_audio(ws: FakeWebSocket, frames: list[bytes]) -> None:
    for frame in frames:
        await ws.push(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode("ascii"),
            }
        )


async def speak_and_endpoint(ws: FakeWebSocket, *, speech_frames: int = 30) -> None:
    frames = [loud_frame() for _ in range(speech_frames)] + [silence_frame() for _ in range(35)]
    await send_audio(ws, frames)


def index_of(ws: FakeWebSocket, event_type: str, response_id: str | None = None) -> int:
    for index, event in enumerate(ws.sent):
        if event.get("type") != event_type:
            continue
        if response_id is None or event.get("response_id") == response_id:
            return index
    raise AssertionError(f"{event_type} not found")


async def test_session_created_then_greeting_is_audio_only():
    ws, session, task = await start_session()
    try:
        await ws.push(
            {
                "type": "speech_text_buffer.append",
                "speech_id": "g1",
                "text": "你来啦，",
            }
        )
        await ws.push(
            {
                "type": "speech_text_buffer.commit",
                "speech_id": "g1",
                "text": "我在呢。",
            }
        )
        started = await ws.wait_for_type("response.output_audio.started")
        await ws.wait_for_type("response.output_audio.delta")
        done = await ws.wait_for_type("response.output_audio.done")
        response_done = await ws.wait_for_type("response.done")
        assert started["response_id"] == done["response_id"] == response_done["response_id"]
        assert index_of(ws, "response.output_audio.started") < index_of(
            ws, "response.output_audio.delta"
        )
        assert index_of(ws, "response.output_audio.done") < index_of(ws, "response.done")
        assert not ws.events("response.output_text.delta")
        assert not ws.events("response.output_text.done")
    finally:
        await stop_session(ws, session, task)


async def test_one_turn_streams_asr_text_audio_in_order():
    ws, session, task = await start_session(mock_asr_text="你好呀，溟月")
    try:
        await speak_and_endpoint(ws)
        completed = await ws.wait_for_type("conversation.item.input_audio_transcription.completed")
        assert completed["transcript"] == "你好呀，溟月"
        await ws.wait_for_type("response.output_text.delta")
        text_done = await ws.wait_for_type("response.output_text.done")
        audio_done = await ws.wait_for_type("response.output_audio.done")
        response_done = await ws.wait_for_type("response.done")
        assert text_done["response_id"] == audio_done["response_id"] == response_done["response_id"]
        # The Android app must keep the reply pending until audio ends so a
        # barge-in can mark it interrupted.
        assert index_of(ws, "response.output_audio.done") < index_of(
            ws, "response.output_text.done"
        )
        assert index_of(ws, "response.output_text.done") < index_of(ws, "response.done")
        assert response_done["status"] == "completed"
    finally:
        await stop_session(ws, session, task)


async def test_empty_asr_returns_to_listening_without_llm():
    ws, session, task = await start_session()
    try:
        # 60 ms speech, below the 200 ms minimum.
        await send_audio(ws, [loud_frame() for _ in range(3)] + [silence_frame() for _ in range(35)])
        failed = await ws.wait_for_type(
            "conversation.item.input_audio_transcription.failed"
        )
        assert failed["error"] == "empty speech"
        await asyncio.sleep(0.1)
        assert not ws.events("conversation.item.input_audio_transcription.completed")
        assert not ws.events("response.output_text.delta")
        assert not ws.events("response.output_audio.started")
    finally:
        await stop_session(ws, session, task)


async def test_cancel_drops_old_generation_audio():
    settings = make_settings(tts_tone_char_ms=120)
    ws, session, task = await start_session(settings=settings)
    try:
        text = "你来啦，我在呢。今天也要开开心心的呀。"
        split = len(text) // 2
        await ws.push(
            {
                "type": "speech_text_buffer.append",
                "speech_id": "g2",
                "text": text[:split],
            }
        )
        await ws.push(
            {
                "type": "speech_text_buffer.commit",
                "speech_id": "g2",
                "text": text[split:],
            }
        )
        started = await ws.wait_for_type("response.output_audio.started")
        old_id = started["response_id"]
        await ws.push({"type": "response.cancel"})
        canceled = await ws.wait_for_type("response.canceled")
        assert canceled["response_id"] == old_id
        canceled_index = index_of(ws, "response.canceled", old_id)
        after = ws.sent[canceled_index + 1 :]
        assert not [
            item
            for item in after
            if item.get("type") == "response.output_audio.delta"
            and item.get("response_id") == old_id
        ]
        assert not [
            item
            for item in after
            if item.get("type") == "response.done"
            and item.get("response_id") == old_id
        ]

        # The session must still be usable for the next turn/greeting.
        await ws.push(
            {
                "type": "speech_text_buffer.append",
                "speech_id": "g3",
                "text": "我在呢。",
            }
        )
        await ws.push(
            {
                "type": "speech_text_buffer.commit",
                "speech_id": "g3",
                "text": "",
            }
        )
        new_done = await ws.wait_for_type("response.done")
        assert new_done["response_id"] != old_id
    finally:
        await stop_session(ws, session, task)


async def test_single_tool_call_round_trip():
    tools = [set_emotion_tool(), ask_expert_tool()]
    ws, session, task = await start_session(mock_asr_text="溟月，笑一个", tools=tools)
    try:
        await speak_and_endpoint(ws)
        tool = await ws.wait_for_type("response.function_call_arguments.done")
        assert len(tool["items"]) == 1
        assert tool["items"][0]["name"] == "set_emotion"
        await ws.push(
            {
                "type": "conversation.item.create",
                "items": [
                    {
                        "type": "message",
                        "role": "tool",
                        "call_id": tool["items"][0]["call_id"],
                        "content": [{"type": "input_text", "text": "ok"}],
                    }
                ],
            }
        )
        await ws.wait_for_type("response.output_text.delta")
        done = await ws.wait_for_type("response.done")
        assert done["status"] == "completed"
        text_done = ws.events("response.output_text.done")[-1]
        assert "笑" in text_done["text"]
    finally:
        await stop_session(ws, session, task)


async def test_multiple_tool_calls_in_one_event():
    tools = [set_emotion_tool(), play_animation_tool(), get_time_tool()]
    ws, session, task = await start_session(mock_asr_text="笑一个，转个圈", tools=tools)
    try:
        await speak_and_endpoint(ws)
        tool = await ws.wait_for_type("response.function_call_arguments.done")
        assert [item["name"] for item in tool["items"]] == [
            "set_emotion",
            "play_animation",
        ]
        await ws.push(
            {
                "type": "conversation.item.create",
                "items": [
                    {
                        "type": "message",
                        "role": "tool",
                        "call_id": item["call_id"],
                        "content": [{"type": "input_text", "text": "ok"}],
                    }
                    for item in tool["items"]
                ],
            }
        )
        await ws.wait_for_type("response.output_text.delta")
        done = await ws.wait_for_type("response.done")
        assert done["status"] == "completed"
    finally:
        await stop_session(ws, session, task)


async def test_session_close_sends_closed_and_stops():
    ws, session, task = await start_session()
    try:
        await ws.push({"type": "session.close"})
        await ws.wait_for_type("session.closed")
        await asyncio.wait_for(task, timeout=2)
        assert ws.closed
        assert session.closed
    finally:
        if not session.closed:
            await stop_session(ws, session, task)


async def test_session_created_echoes_resolved_voice(tmp_path):
    registry = make_voice_registry(tmp_path)
    ws, session, task = await start_session(
        voice_registry=registry, voice="mingyue_xixi_v1"
    )
    try:
        created = ws.events("session.created")[0]["session"]
        updated = ws.events("session.updated")[0]["session"]
        assert created["voice"] == "mingyue_xixi_v1"
        assert created["voice_name"] == "汐汐"
        assert created["voice_fallback"] is False
        assert updated["voice"] == "mingyue_xixi_v1"
        assert updated["voice_name"] == "汐汐"
        assert updated["voice_fallback"] is False
        assert session._voice == "mingyue_xixi_v1"
    finally:
        await stop_session(ws, session, task)


async def test_unknown_voice_falls_back_to_default(tmp_path):
    registry = make_voice_registry(tmp_path)
    ws, session, task = await start_session(
        voice_registry=registry, voice="unknown_voice"
    )
    try:
        created = ws.events("session.created")[0]["session"]
        assert created["voice"] == registry.default_id
        assert created["voice_name"] == "月月"
        assert created["voice_fallback"] is True
        assert session._voice == registry.default_id
    finally:
        await stop_session(ws, session, task)


async def test_empty_voice_uses_default_without_fallback(tmp_path):
    registry = make_voice_registry(tmp_path)
    ws, session, task = await start_session(voice_registry=registry, voice="")
    try:
        created = ws.events("session.created")[0]["session"]
        assert created["voice"] == registry.default_id
        assert created["voice_fallback"] is False
    finally:
        await stop_session(ws, session, task)


async def test_absent_voice_uses_default_without_fallback(tmp_path):
    registry = make_voice_registry(tmp_path)
    ws, session, task = await start_session(
        voice_registry=registry, include_voice=False
    )
    try:
        created = ws.events("session.created")[0]["session"]
        assert created["voice"] == registry.default_id
        assert created["voice_fallback"] is False
    finally:
        await stop_session(ws, session, task)
