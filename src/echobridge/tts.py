"""TTS adapters.

Phase 1 mock: deterministic tones or a local WAV.
Phase 3: Qwen3-TTS worker over localhost HTTP, sentence-level streaming.
"""

from __future__ import annotations

import asyncio
import base64
import os
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import httpx

from .audio import load_wav_mono_24k, silence_pcm16, tone_pcm16
from .config import Settings
from .logging import get_logger

logger = get_logger("echobridge.tts")


class TTSBackend(Protocol):
    def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]: ...

    def cached(self, text: str, voice_id: str) -> bytes | None: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...


class MockTTS:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._first_chunk_delay = max(0, settings.tts_first_chunk_ms) / 1000.0
        self._wav: bytes | None = None
        if settings.mock_tts_wav:
            self._wav = load_wav_mono_24k(settings.mock_tts_wav)

    async def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]:
        if not text.strip():
            return
        if self._first_chunk_delay:
            await asyncio.sleep(self._first_chunk_delay)
        if self._wav is not None:
            pcm = self._wav
        else:
            pcm = tone_pcm16(
                text,
                sample_rate=24_000,
                char_ms=self._settings.tts_tone_char_ms,
            )
        pcm += silence_pcm16(60, 24_000)
        block = 4_800
        for offset in range(0, len(pcm), block):
            chunk = pcm[offset : offset + block]
            if chunk:
                yield chunk

    def cached(self, text: str, voice_id: str) -> bytes | None:
        if not text.strip():
            return None
        # Mock TTS has no model latency; synthesizing a short filler is cheap.
        return tone_pcm16(
            text,
            sample_rate=24_000,
            char_ms=self._settings.tts_tone_char_ms,
        )

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


class Qwen3TTSClient:
    """Sentence-level streaming client for the local Qwen3-TTS worker."""

    def __init__(self, settings: Settings) -> None:
        self._voice_id = settings.tts_voice_id
        self._backchannel_phrases = (
            settings.backchannel_phrases if settings.backchannel_enabled else ()
        )
        self._client = httpx.AsyncClient(
            base_url=settings.tts_url,
            timeout=httpx.Timeout(settings.tts_timeout_s, connect=2.0),
        )
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._backchannel_cache: dict[tuple[str, str], bytes] = {}
        self._cache_max = 8

    def cached(self, text: str, voice_id: str) -> bytes | None:
        key = (voice_id or self._voice_id, text.strip())
        pinned = self._backchannel_cache.get(key)
        return pinned if pinned is not None else self._cache.get(key)

    async def _fetch(self, text: str, voice: str, generation: int = 0) -> bytes:
        key = (voice, text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        response = await self._client.post(
            "/v1/synthesize",
            json={"text": text, "voice_id": voice, "generation": generation},
        )
        response.raise_for_status()
        pcm = response.content
        if not pcm:
            raise RuntimeError("TTS worker returned empty audio")
        self._cache[key] = pcm
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return pcm

    async def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        voice = voice_id or self._voice_id
        pcm = await self._fetch(text, voice, generation)
        block = 4_800  # 100 ms @ 24 kHz
        for offset in range(0, len(pcm), block):
            chunk = pcm[offset : offset + block]
            if chunk:
                yield chunk

    async def warmup(self) -> None:
        await self._client.get("/healthz")
        for phrase in self._backchannel_phrases:
            try:
                pcm = await self._fetch(phrase, self._voice_id)
                if pcm:
                    self._backchannel_cache[(self._voice_id, phrase)] = pcm
            except Exception as exc:
                logger.warning("TTS backchannel prewarm failed phrase=%r: %s", phrase, exc)

    async def close(self) -> None:
        await self._client.aclose()


class VLLMTTSClient:
    """Streaming client for the vLLM-Omni Qwen3-TTS Base server.

    The server exposes an OpenAI-compatible /v1/audio/speech endpoint with
    raw PCM streaming.  This is the low-latency runtime for Phase 3; the HF
    worker remains available as a reference implementation.
    """

    def __init__(self, settings: Settings) -> None:
        self._voice_id = settings.tts_voice_id
        self._backchannel_phrases = (
            settings.backchannel_phrases if settings.backchannel_enabled else ()
        )
        ref_audio = Path(os.path.expanduser(settings.tts_ref_audio))
        ref_text = Path(os.path.expanduser(settings.tts_ref_text))
        if not ref_audio.exists():
            raise FileNotFoundError(f"TTS reference audio missing: {ref_audio}")
        if not ref_text.exists():
            raise FileNotFoundError(f"TTS reference text missing: {ref_text}")
        self._ref_text = ref_text.read_text(encoding="utf-8").strip()
        if not self._ref_text:
            raise ValueError("TTS reference text is empty")
        self._ref_audio = (
            "data:audio/wav;base64,"
            + base64.b64encode(ref_audio.read_bytes()).decode("ascii")
        )
        self._client = httpx.AsyncClient(
            base_url=settings.tts_url,
            timeout=httpx.Timeout(settings.tts_timeout_s, connect=2.0),
        )
        self._api_key = settings.tts_api_key
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._backchannel_cache: dict[tuple[str, str], bytes] = {}
        self._cache_max = 8

    def _payload(self, text: str, voice: str) -> dict:
        return {
            "input": text,
            "task_type": "Base",
            "language": "Chinese",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "ref_audio": self._ref_audio,
            "ref_text": self._ref_text,
        }

    async def _stream(self, text: str, voice: str) -> AsyncIterator[bytes]:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with self._client.stream(
            "POST", "/v1/audio/speech", json=self._payload(text, voice), headers=headers
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode("utf-8", "replace")
                raise RuntimeError(f"vLLM TTS HTTP {response.status_code}: {detail[:300]}")
            leftover = b""
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                data = leftover + chunk
                usable = len(data) - (len(data) % 2)
                leftover = data[usable:]
                if usable:
                    yield data[:usable]
            if leftover:
                yield leftover + bytes(1)

    async def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]:
        text = text.strip()
        if not text:
            return
        voice = voice_id or self._voice_id
        key = (voice, text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            for offset in range(0, len(cached), 4800):
                yield cached[offset : offset + 4800]
            return
        collected = bytearray()
        async for chunk in self._stream(text, voice):
            collected.extend(chunk)
            yield chunk
        if collected:
            self._cache[key] = bytes(collected)
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)

    def cached(self, text: str, voice_id: str) -> bytes | None:
        key = (voice_id or self._voice_id, text.strip())
        pinned = self._backchannel_cache.get(key)
        return pinned if pinned is not None else self._cache.get(key)

    async def _prefetch(self, text: str, voice: str) -> bytes:
        key = (voice, text)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        collected = bytearray()
        async for chunk in self._stream(text, voice):
            collected.extend(chunk)
        pcm = bytes(collected)
        if pcm:
            self._cache[key] = pcm
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return pcm

    async def warmup(self) -> None:
        try:
            await self._client.get("/health")
        except Exception:
            pass
        # Pre-synthesize the fixed app greeting; this is silent and warms the
        # talker/codec path so the first real greeting is cache-hit fast.
        greeting = "你来啦，我在呢。"
        try:
            await self._prefetch(greeting, self._voice_id)
        except Exception as exc:
            logger.warning("TTS greeting prewarm failed: %s", exc)
        for phrase in self._backchannel_phrases:
            try:
                pcm = await self._prefetch(phrase, self._voice_id)
                if pcm:
                    self._backchannel_cache[(self._voice_id, phrase)] = pcm
            except Exception as exc:
                logger.warning("TTS backchannel prewarm failed phrase=%r: %s", phrase, exc)

    async def close(self) -> None:
        await self._client.aclose()


def make_tts(settings: Settings) -> TTSBackend:
    if settings.tts_backend == "vllm":
        return VLLMTTSClient(settings)
    if settings.tts_backend == "qwen3":
        return Qwen3TTSClient(settings)
    return MockTTS(settings)
