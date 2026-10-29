from __future__ import annotations

from collections.abc import AsyncIterator

from echobridge.session import Session
from echobridge.text import pop_tts_chunk, split_for_tts
from tests.fakes import FakeWebSocket
from tests.helpers import make_settings, make_voice_registry


def simulate_stream(text: str) -> list[str]:
    """Feed one character at a time, exactly like a streaming LLM delta."""
    buffer = ""
    chunks: list[str] = []
    first = True
    for char in text:
        buffer += char
        while True:
            chunk, buffer = pop_tts_chunk(buffer, first_chunk=first)
            if not chunk:
                break
            chunks.append(chunk)
            first = False
    if buffer.strip():
        chunks.extend(split_for_tts(buffer))
    return chunks


def test_first_chunk_splits_at_clause_and_keeps_tail():
    chunk, remaining = pop_tts_chunk(
        "好的没问题，我们继续聊天吧", first_chunk=True
    )
    assert chunk == "好的没问题，"
    assert remaining == "我们继续聊天吧"


def test_first_chunk_soft_cap_keeps_old_latency():
    text = "一二三四五六七八九十十一十二十三十四十五"
    chunk, remaining = pop_tts_chunk(text, first_chunk=True)
    assert 4 <= len(chunk) <= 15
    assert remaining


def test_first_chunk_does_not_split_supercute():
    text = "你好呀，我今天看到了一只超级可爱的小鲸鱼"
    chunk, remaining = pop_tts_chunk(text, first_chunk=True)
    assert "超级可" not in chunk
    assert chunk.endswith("了")
    assert remaining.startswith("一只超级可爱")


def test_subsequent_clause_split_keeps_unfinished_tail():
    text = "它包含了很多很多的内容，后面还有没有说完的残句，需要继续拼接下去"
    chunk, remaining = pop_tts_chunk(text, first_chunk=False)
    assert chunk.endswith("残句，")
    assert remaining == "需要继续拼接下去"


def test_no_punctuation_hard_cap_uses_function_word_boundary():
    text = (
        "我们一起去公园散步然后回家的路上看到了很多花"
        "和很多可爱的小动物还有蓝色的天空"
    )
    chunk, remaining = pop_tts_chunk(text, first_chunk=False)
    assert chunk.endswith(tuple("的了吗呢吧呀啊和与在是有就都而但着过地得"))
    assert remaining == "天空"
    assert "超级可" not in chunk


def test_long_ascii_token_is_not_broken():
    token = "x" * 50
    assert pop_tts_chunk(token, first_chunk=False) == ("", token)
    chunk, remaining = pop_tts_chunk(token + " y", first_chunk=False)
    assert chunk.strip() == token
    assert remaining.strip() == "y"


def test_streaming_mixed_text_never_breaks_words_or_decimals():
    text = (
        "你好呀，我今天看到了一只超级可爱的小鲸鱼，"
        "它 really likes 3.14 and OpenAI models，真的很可爱。"
    )
    chunks = simulate_stream(text)
    joined = "".join(chunks)
    assert chunks
    assert "超级可爱" in joined
    assert "3.14" in joined
    assert "OpenAI" in joined
    assert not any(chunk.endswith("超级可") for chunk in chunks)
    # A comma split must not send the half-clause after it in the same chunk.
    assert not any(chunk.endswith("小鲸鱼，它") for chunk in chunks)


def test_split_for_tts_preserves_complete_words_on_final_flush():
    text = "你好呀，我今天看到了一只超级可爱的小鲸鱼，它真的很可爱。"
    chunks = split_for_tts(text)
    assert chunks
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")
    assert any("超级可爱" in chunk for chunk in chunks)
    assert not any(chunk.endswith("超级可") for chunk in chunks)


class RecordingTTS:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]:
        self.texts.append(text)
        yield b"\x00\x00" * 2400

    def cached(self, text: str, voice_id: str) -> bytes | None:
        return None

    async def warmup(self) -> None:
        return None

    async def ensure_warm(self, voice_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


async def test_session_cleans_markdown_in_streamed_chunks(tmp_path):
    registry = make_voice_registry(tmp_path)
    tts = RecordingTTS()
    session = Session(
        FakeWebSocket(),
        make_settings(tts_backend="mock", tts_voice_dir=str(tmp_path)),
        tts=tts,
        voice_registry=registry,
    )
    text = "**你好**，我今天看到了一只超级可爱的小鲸鱼。"

    async def source() -> AsyncIterator[str]:
        for char in text:
            yield char

    ctx = session._begin_response()
    await session._stream_speech(ctx, source(), emit_text=False)
    assert tts.texts
    assert all("*" not in chunk for chunk in tts.texts)
    assert any("超级可爱" in chunk for chunk in tts.texts)


async def test_session_streaming_feeds_only_safe_chunks(tmp_path):
    registry = make_voice_registry(tmp_path)
    tts = RecordingTTS()
    session = Session(
        FakeWebSocket(),
        make_settings(tts_backend="mock", tts_voice_dir=str(tmp_path)),
        tts=tts,
        voice_registry=registry,
    )
    text = "你好呀，我今天看到了一只超级可爱的小鲸鱼，它真的很可爱。"

    async def source() -> AsyncIterator[str]:
        for char in text:
            yield char

    ctx = session._begin_response()
    await session._stream_speech(ctx, source(), emit_text=False)

    assert tts.texts
    assert any("超级可爱" in chunk for chunk in tts.texts)
    assert not any(chunk.endswith("超级可") for chunk in tts.texts)
    # Every emitted chunk is non-empty and the final tail is flushed.
    assert all(chunk.strip() for chunk in tts.texts)
    assert "".join(tts.texts).replace(" ", "") == text.replace(" ", "")
