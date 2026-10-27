from __future__ import annotations

from echobridge.text import is_stop_command
from tests.helpers import make_settings
from tests.test_session import speak_and_endpoint, start_session, stop_session


def test_stop_command_detection():
    for text in ("停", "停一下", "停下", "别说了", "不用了", "算了", "等等", "stop", "Wait!"):
        assert is_stop_command(text), text
    for text in ("下，不用讲了。", "一，不用讲了。", "好，别说了。", "停一下，不用讲了。"):
        assert is_stop_command(text), text
    for text in ("停车", "停下来讲个笑话", "现在几点了", "不用谢", ""):
        assert not is_stop_command(text), text


async def test_stop_command_does_not_start_a_turn():
    settings = make_settings(backchannel_enabled=True, backchannel_delay_ms=50)
    ws, session, task = await start_session(
        mock_asr_text="停一下", settings=settings
    )
    try:
        await speak_and_endpoint(ws)
        await ws.wait_for_type("conversation.item.input_audio_transcription.completed")
        done = await ws.wait_for_type("response.done")
        assert done["status"] == "stopped"
        assert not ws.events("response.output_text.delta")
        assert not ws.events("response.output_audio.started")
        assert not ws.events("response.backchannel")
    finally:
        await stop_session(ws, session, task)
