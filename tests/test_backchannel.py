from __future__ import annotations

import asyncio
import base64

from tests.helpers import make_settings, set_emotion_tool
from tests.test_session import (
    index_of,
    speak_and_endpoint,
    start_session,
    stop_session,
)


async def test_backchannel_precedes_slow_formal_audio():
    settings = make_settings(
        backchannel_enabled=True,
        backchannel_delay_ms=50,
        tts_first_chunk_ms=800,
    )
    ws, session, task = await start_session(settings=settings)
    try:
        await speak_and_endpoint(ws)
        backchannel = await ws.wait_for_type("response.backchannel")
        await ws.wait_for_type("response.output_audio.started")
        assert backchannel["response_id"].startswith("resp_")
        assert backchannel["sound_id"] in settings.backchannel_phrases
        assert base64.b64decode(backchannel["delta"])
        assert index_of(ws, "response.backchannel") < index_of(
            ws, "response.output_audio.started"
        )
        # The formal response still completes normally.
        await ws.wait_for_type("response.done")
    finally:
        await stop_session(ws, session, task)


async def test_backchannel_not_sent_when_formal_audio_is_fast():
    settings = make_settings(
        backchannel_enabled=True,
        backchannel_delay_ms=200,
        tts_first_chunk_ms=1,
    )
    ws, session, task = await start_session(settings=settings)
    try:
        await speak_and_endpoint(ws)
        await ws.wait_for_type("response.done")
        assert not ws.events("response.backchannel")
    finally:
        await stop_session(ws, session, task)


async def test_backchannel_does_not_repeat_across_turns():
    settings = make_settings(
        backchannel_enabled=True,
        backchannel_delay_ms=30,
        tts_first_chunk_ms=400,
    )
    ws, session, task = await start_session(settings=settings)
    try:
        await speak_and_endpoint(ws)
        first = await ws.wait_for_type("response.backchannel")
        await ws.wait_for_type("response.done")
        first_response = first["response_id"]

        # Second turn.  Reset the event list so wait_for_type finds the new one.
        ws.sent.clear()
        await speak_and_endpoint(ws)
        second = await ws.wait_for_type("response.backchannel")
        assert second["response_id"] != first_response
        assert second["sound_id"] != first["sound_id"]
    finally:
        await stop_session(ws, session, task)


async def test_one_backchannel_max_per_turn_with_tool_hop():
    settings = make_settings(
        backchannel_enabled=True,
        backchannel_delay_ms=30,
        tts_first_chunk_ms=400,
    )
    ws, session, task = await start_session(
        mock_asr_text="溟月，笑一个",
        tools=[set_emotion_tool()],
        settings=settings,
    )
    try:
        await speak_and_endpoint(ws)
        tool = await ws.wait_for_type("response.function_call_arguments.done")
        call_id = tool["items"][0]["call_id"]
        await ws.push(
            {
                "type": "conversation.item.create",
                "items": [
                    {
                        "type": "message",
                        "role": "tool",
                        "call_id": call_id,
                        "content": [{"type": "input_text", "text": "ok"}],
                    }
                ],
            }
        )
        await ws.wait_for_type("response.done")
        assert len(ws.events("response.backchannel")) == 1
    finally:
        await stop_session(ws, session, task)

