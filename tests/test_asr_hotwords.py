from __future__ import annotations

import base64

from echobridge.asr import encode_asr_context, is_hotword_echo
from echobridge.config import Settings

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
