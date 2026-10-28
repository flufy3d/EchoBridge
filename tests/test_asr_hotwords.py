from __future__ import annotations

import asyncio
import base64

from echobridge.asr import (
    ASRResult,
    encode_asr_context,
    is_hotword_echo,
    parse_hotwords,
)
from echobridge.config import Settings, build_asr_context
from echobridge.session import Session
from tests.fakes import FakeWebSocket
from tests.helpers import loud_frame, make_settings, session_create, silence_frame

HOTWORDS = ("ChatGPT", "Claude", "DeepSeek", "豆包", "溟月", "小鲸鱼", "大肥鱼")


def test_hotwords_build_context(monkeypatch):
    monkeypatch.setenv("ECHOBRIDGE_DEV", "1")
    monkeypatch.setenv("ECHOBRIDGE_ASR_HOTWORDS", "|".join(HOTWORDS))
    settings = Settings.from_env()
    assert settings.asr_hotwords == HOTWORDS
    assert settings.asr_context == (
        "Technical terms: ChatGPT, Claude, DeepSeek, 豆包, 溟月, 小鲸鱼, 大肥鱼"
    )


def test_no_hotwords_means_no_context():
    assert Settings().asr_context == ""
    assert build_asr_context([]) == ""


def test_session_hotword_parsing_accepts_array_and_delimiters():
    assert parse_hotwords(["ChatGPT", " 溟月 "]) == ("ChatGPT", "溟月")
    assert parse_hotwords("ChatGPT,Claude|溟月|小鲸鱼") == (
        "ChatGPT",
        "Claude",
        "溟月",
        "小鲸鱼",
    )
    assert parse_hotwords(None) is None


def test_context_header_is_utf8_base64():
    encoded = encode_asr_context("溟月")
    assert base64.b64decode(encoded).decode("utf-8") == "溟月"


def test_full_hotword_list_echo_is_detected():
    assert is_hotword_echo(
        "Technical terms: ChatGPT, Claude, DeepSeek, 豆包, 溟月, 小鲸鱼, 大肥鱼",
        HOTWORDS,
    )
    assert is_hotword_echo(
        "ChatGPT Claude DeepSeek 豆包 溟月 小鲸鱼 大肥鱼。",
        HOTWORDS,
    )


def test_normal_speech_is_not_detected_as_echo():
    assert not is_hotword_echo("溟月", HOTWORDS)
    assert not is_hotword_echo("溟月，今天天气怎么样", HOTWORDS)
    assert not is_hotword_echo("帮我比较一下 ChatGPT 和 Claude", HOTWORDS)
    assert not is_hotword_echo("小鲸鱼和大肥鱼都在吗", HOTWORDS)


class _RecordingASR:
    def __init__(self) -> None:
        self.contexts: list[str | None] = []

    async def transcribe(self, pcm, sample_rate=16000, context=None):
        self.contexts.append(context)
        return ASRResult("你好呀，溟月", "zh", 0.0)

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


async def _run_session_context(event: dict) -> str | None:
    ws = FakeWebSocket()
    asr = _RecordingASR()
    session = Session(
        ws,
        make_settings(asr_hotwords=("网关默认词",)),
        device_id="test-device",
        asr=asr,
    )
    task = asyncio.create_task(session.run())
    try:
        await ws.push(event)
        await ws.wait_for_type("session.created")

        for _ in range(30):
            await ws.push(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(loud_frame()).decode("ascii"),
                }
            )
        for _ in range(35):
            await ws.push(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(silence_frame()).decode("ascii"),
                }
            )

        await ws.wait_for_type(
            "conversation.item.input_audio_transcription.completed"
        )
        return asr.contexts[-1]
    finally:
        await session.close("test")
        await asyncio.wait_for(task, timeout=2)


async def test_session_create_hotwords_override_gateway_default():
    event = session_create()
    event["extension"]["asr"]["extra"]["hotwords"] = ["ChatGPT", "溟月"]
    assert await _run_session_context(event) == "Technical terms: ChatGPT, 溟月"


async def test_session_create_without_hotwords_uses_gateway_default():
    # None tells Qwen3ASRClient to use the gateway-wide ECHOBRIDGE_ASR_HOTWORDS.
    assert await _run_session_context(session_create()) is None


async def test_session_create_empty_hotwords_disables_gateway_default():
    event = session_create()
    event["extension"]["asr"]["extra"]["hotwords"] = []
    assert await _run_session_context(event) == ""


async def test_echobridge_ignores_doubao_boosting_table_id():
    event = session_create()
    event["extension"]["asr"]["extra"] = {"boosting_table_id": "tbl_123"}
    # EchoBridge has no weighted table: leave the gateway-wide default intact.
    assert await _run_session_context(event) is None
