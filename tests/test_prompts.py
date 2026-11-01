from __future__ import annotations

import asyncio

from echobridge.config import Settings
from echobridge.prompts import (
    NEUTRAL_INSTRUCTIONS,
    VOICE_RULES,
    build_system_prompt,
    instructions_fingerprint,
)
from echobridge.session import Session
from tests.fakes import FakeWebSocket
from tests.helpers import make_settings, session_create

CLIENT_PERSONA = (
    "你是「测试甲」，一只橘猫。"
    + "你有一身橘色虎斑毛，穿白色花边袜和深蓝小皮鞋。" * 90
    + "被问到腿时回答『喵，我有四条腿』。"
)


def test_client_instructions_are_verbatim_and_not_truncated():
    assert len(CLIENT_PERSONA) > 2000
    prompt = build_system_prompt(CLIENT_PERSONA)
    assert prompt.startswith(CLIENT_PERSONA)
    assert CLIENT_PERSONA in prompt
    assert prompt == f"{CLIENT_PERSONA}\n\n{VOICE_RULES}"


def test_whitespace_inside_instructions_is_preserved():
    text = "  测试人格  \n第二行\n"
    prompt = build_system_prompt(text)
    assert prompt.startswith(text)
    assert prompt != f"{text.strip()}\n\n{VOICE_RULES}"


def test_empty_instructions_use_neutral_prompt_not_a_character():
    for empty in ("", "   ", "\n\t"):
        prompt = build_system_prompt(empty)
        assert prompt.startswith(NEUTRAL_INSTRUCTIONS)
        for forbidden in ("溟月", "小鲸鱼", "鲸鱼", "月月"):
            assert forbidden not in prompt


def test_voice_rules_contain_no_persona():
    for forbidden in ("溟月", "小鲸鱼", "鲸鱼", "月月", "尾巴", "深蓝", "名字"):
        assert forbidden not in VOICE_RULES


def test_fingerprint_matches_sha256_prefix():
    chars, raw_bytes, digest = instructions_fingerprint("abc")
    assert (chars, raw_bytes, digest) == (3, 3, "ba7816bf")
    chars, raw_bytes, digest = instructions_fingerprint("溟月")
    assert chars == 2
    assert raw_bytes == 6
    assert len(digest) == 8


async def _start(
    *,
    instructions: str,
    settings: Settings | None = None,
    extra_extension: bool = False,
) -> tuple[FakeWebSocket, Session, asyncio.Task[None]]:
    ws = FakeWebSocket()
    session = Session(
        ws,
        settings or make_settings(),
        device_id="test-device",
        mock_asr_text="你好呀",
    )
    task = asyncio.create_task(session.run())
    event = session_create([], instructions=instructions)
    if extra_extension:
        event["extension"]["asr"]["extra"]["future_field"] = {"nested": [1, 2, 3]}
        event["extension"]["brand_new_block"] = {"anything": True}
        event["session"]["future_session_field"] = {"ignored": True}
    await ws.push(event)
    await ws.wait_for_type("session.created")
    return ws, session, task


async def _stop(ws: FakeWebSocket, session: Session, task: asyncio.Task[None]) -> None:
    await session.close("test")
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_session_keeps_instructions_with_unknown_extension_fields():
    ws, session, task = await _start(
        instructions=CLIENT_PERSONA, extra_extension=True
    )
    try:
        assert session._instructions == CLIENT_PERSONA
        system = session._build_llm_messages()[0]
        assert system == {
            "role": "system",
            "content": build_system_prompt(CLIENT_PERSONA),
        }
    finally:
        await _stop(ws, session, task)


async def test_instructions_are_per_session_and_never_cached():
    ws1, session1, task1 = await _start(instructions="第一段人格")
    try:
        assert "第一段人格" in session1._build_llm_messages()[0]["content"]
    finally:
        await _stop(ws1, session1, task1)

    ws2, session2, task2 = await _start(instructions="")
    try:
        system = session2._build_llm_messages()[0]["content"]
        assert system.startswith(NEUTRAL_INSTRUCTIONS)
        assert "第一段人格" not in system
    finally:
        await _stop(ws2, session2, task2)


async def test_unexpected_instructions_shape_uses_neutral_prompt():
    ws = FakeWebSocket()
    session = Session(
        ws, make_settings(), device_id="test-device", mock_asr_text="你好呀"
    )
    task = asyncio.create_task(session.run())
    event = session_create([])
    event["session"]["instructions"] = {"text": "不是字符串"}
    await ws.push(event)
    await ws.wait_for_type("session.created")
    try:
        assert session._instructions == ""
        system = session._build_llm_messages()[0]["content"]
        assert system.startswith(NEUTRAL_INSTRUCTIONS)
        assert "不是字符串" not in system
    finally:
        await _stop(ws, session, task)


async def test_prompt_dump_is_exact(tmp_path):
    settings = make_settings(prompt_dump_dir=str(tmp_path))
    ws, session, task = await _start(
        instructions=CLIENT_PERSONA, settings=settings
    )
    try:
        dump = (tmp_path / f"{session.session_id}.txt").read_text(encoding="utf-8")
        assert dump == build_system_prompt(CLIENT_PERSONA)
    finally:
        await _stop(ws, session, task)
