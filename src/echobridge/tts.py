"""TTS adapters.

Phase 1 mock: deterministic tones or a local WAV.
Phase 3: Qwen3-TTS worker over localhost HTTP, sentence-level streaming.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import zlib
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import httpx

from .audio import (
    load_wav_mono_24k,
    silence_pcm16,
    tone_pcm16,
    trim_silence_pcm16,
)
from .config import Settings
from .logging import get_logger
from .voices import VoiceRegistry, VoiceSpec, load_voice_registry

logger = get_logger("echobridge.tts")

GREETING = "你来啦，我在呢。"


def load_backchannel_clips(
    settings: Settings, registry: VoiceRegistry
) -> dict[tuple[str, str], bytes]:
    """Load hand-picked backchannel recordings, keyed by (voice_id, phrase).

    Layout: ``<clip_dir>/<voice_id>/clips.json`` mapping phrase -> WAV path
    (relative to that file).  Only configured phrases are loaded; anything
    missing or unreadable falls back to TTS synthesis at warmup.
    """
    if not settings.backchannel_enabled or not settings.backchannel_phrases:
        return {}
    root = Path(
        os.path.expanduser(
            settings.backchannel_clip_dir
            or os.path.join(settings.tts_voice_dir, "backchannel")
        )
    )
    clips: dict[tuple[str, str], bytes] = {}
    for spec in registry.voices:
        manifest = root / spec.id / "clips.json"
        if not manifest.is_file():
            continue
        try:
            entries = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("backchannel clips unreadable %s: %s", manifest, exc)
            continue
        if not isinstance(entries, dict):
            logger.warning("backchannel clips must be an object: %s", manifest)
            continue
        for phrase in settings.backchannel_phrases:
            name = entries.get(phrase)
            if not isinstance(name, str) or not name:
                continue
            try:
                pcm = trim_silence_pcm16(load_wav_mono_24k(manifest.parent / name))
            except Exception as exc:
                logger.warning(
                    "backchannel clip failed voice=%s phrase=%r: %s",
                    spec.id,
                    phrase,
                    exc,
                )
                continue
            if pcm:
                clips[(spec.id, phrase)] = pcm
        logger.info(
            "backchannel clips voice=%s loaded=%d",
            spec.id,
            sum(1 for voice, _ in clips if voice == spec.id),
        )
    return clips


class TTSBackend(Protocol):
    def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]: ...

    def cached(self, text: str, voice_id: str) -> bytes | None: ...

    async def warmup(self) -> None: ...

    async def ensure_warm(self, voice_id: str) -> None: ...

    async def close(self) -> None: ...


class MockTTS:
    def __init__(
        self, settings: Settings, registry: VoiceRegistry | None = None
    ) -> None:
        self._settings = settings
        self._registry = registry or load_voice_registry(settings)
        self._first_chunk_delay = max(0, settings.tts_first_chunk_ms) / 1000.0
        self._wav: bytes | None = None
        if settings.mock_tts_wav:
            self._wav = load_wav_mono_24k(settings.mock_tts_wav)

    def _note_offset(self, voice_id: str) -> int:
        spec, _ = self._registry.resolve(voice_id)
        # Shift the pentatonic sequence per voice so mock sessions can be
        # distinguished by ear (and by byte comparison in tests).
        return (zlib.crc32(spec.id.encode("utf-8")) >> 3) % 5

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
                note_offset=self._note_offset(voice_id),
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
            note_offset=self._note_offset(voice_id),
        )

    async def warmup(self) -> None:
        return None

    async def ensure_warm(self, voice_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


class Qwen3TTSClient:
    """Sentence-level streaming client for the local Qwen3-TTS worker."""

    def __init__(
        self, settings: Settings, registry: VoiceRegistry | None = None
    ) -> None:
        self._registry = registry or load_voice_registry(settings)
        self._voice_id = self._registry.default_id
        self._backchannel_phrases = (
            settings.backchannel_phrases if settings.backchannel_enabled else ()
        )
        self._client = httpx.AsyncClient(
            base_url=settings.tts_url,
            timeout=httpx.Timeout(settings.tts_timeout_s, connect=2.0),
        )
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._backchannel_cache: dict[tuple[str, str], bytes] = (
            load_backchannel_clips(settings, self._registry)
        )
        self._cache_max = 8
        self._warm_locks = {
            spec.id: asyncio.Lock() for spec in self._registry.voices
        }
        self._warmed: set[str] = set()
        self._warm_tasks: set[asyncio.Task[None]] = set()

    def _resolve_voice(self, voice_id: str | None) -> str:
        return self._registry.resolve(voice_id)[0].id

    def cached(self, text: str, voice_id: str) -> bytes | None:
        key = (self._resolve_voice(voice_id), text.strip())
        pinned = self._backchannel_cache.get(key)
        return pinned if pinned is not None else self._cache.get(key)

    async def _fetch(self, text: str, voice: str, generation: int = 0) -> bytes:
        voice = self._resolve_voice(voice)
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
        pcm = await self._fetch(text, self._resolve_voice(voice_id), generation)
        block = 4_800  # 100 ms @ 24 kHz
        for offset in range(0, len(pcm), block):
            chunk = pcm[offset : offset + block]
            if chunk:
                yield chunk

    async def _warm_voice(self, voice: str) -> None:
        voice = self._resolve_voice(voice)
        if voice in self._warmed:
            return
        lock = self._warm_locks.setdefault(voice, asyncio.Lock())
        async with lock:
            if voice in self._warmed:
                return
            try:
                pcm = await self._fetch(GREETING, voice)
                if pcm:
                    self._backchannel_cache[(voice, GREETING)] = pcm
            except Exception as exc:
                logger.warning("TTS greeting prewarm failed voice=%s: %s", voice, exc)
            for phrase in self._backchannel_phrases:
                if (voice, phrase) in self._backchannel_cache:
                    continue  # hand-picked clip
                try:
                    pcm = trim_silence_pcm16(await self._fetch(phrase, voice))
                    if pcm:
                        self._backchannel_cache[(voice, phrase)] = pcm
                except Exception as exc:
                    logger.warning(
                        "TTS backchannel prewarm failed voice=%s phrase=%r: %s",
                        voice,
                        phrase,
                        exc,
                    )
            self._warmed.add(voice)

    def _schedule_background_warm(self) -> None:
        for spec in self._registry.voices:
            if spec.id in self._warmed:
                continue
            task = asyncio.create_task(
                self._warm_voice(spec.id), name=f"tts-warm-{spec.id}"
            )
            self._warm_tasks.add(task)
            task.add_done_callback(self._warm_tasks.discard)

    async def warmup(self) -> None:
        await self._client.get("/healthz")
        await self._warm_voice(self._voice_id)
        self._schedule_background_warm()

    async def ensure_warm(self, voice_id: str) -> None:
        await self._warm_voice(voice_id)

    async def close(self) -> None:
        tasks = list(self._warm_tasks)
        self._warm_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._client.aclose()


class VLLMTTSClient:
    """Streaming client for the vLLM-Omni Qwen3-TTS Base server.

    The server exposes an OpenAI-compatible /v1/audio/speech endpoint with
    raw PCM streaming.  This is the low-latency runtime for Phase 3; the HF
    worker remains available as a reference implementation.
    """

    def __init__(
        self, settings: Settings, registry: VoiceRegistry | None = None
    ) -> None:
        self._registry = registry or load_voice_registry(settings)
        self._voice_id = self._registry.default_id
        self._backchannel_phrases = (
            settings.backchannel_phrases if settings.backchannel_enabled else ()
        )
        self._voice_refs: dict[str, tuple[str, str]] = {}
        for spec in self._registry.voices:
            try:
                self._voice_refs[spec.id] = self._encode_ref(spec)
            except Exception as exc:
                if spec.id == self._registry.default_id:
                    raise
                logger.warning(
                    "TTS reference unavailable for voice %s: %s", spec.id, exc
                )
        if self._registry.default_id not in self._voice_refs:
            raise FileNotFoundError(
                f"TTS reference audio missing for default voice "
                f"{self._registry.default_id}"
            )
        self._client = httpx.AsyncClient(
            base_url=settings.tts_url,
            timeout=httpx.Timeout(settings.tts_timeout_s, connect=2.0),
        )
        self._api_key = settings.tts_api_key
        self._cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._backchannel_cache: dict[tuple[str, str], bytes] = (
            load_backchannel_clips(settings, self._registry)
        )
        self._cache_max = 8
        self._warm_locks = {
            spec.id: asyncio.Lock() for spec in self._registry.voices
        }
        self._warmed: set[str] = set()
        self._warm_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _encode_ref(spec: VoiceSpec) -> tuple[str, str]:
        ref_audio = Path(os.path.expanduser(str(spec.ref_audio)))
        if not ref_audio.is_file():
            raise FileNotFoundError(f"TTS reference audio missing: {ref_audio}")
        if not spec.ref_text:
            raise ValueError("TTS reference text is empty")
        encoded = base64.b64encode(ref_audio.read_bytes()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}", spec.ref_text

    def _resolve_voice(self, voice_id: str | None) -> str:
        return self._registry.resolve(voice_id)[0].id

    def _payload(self, text: str, voice: str) -> dict:
        spec, _ = self._registry.resolve(voice)
        ref = self._voice_refs.get(spec.id) or self._voice_refs[
            self._registry.default_id
        ]
        return {
            "input": text,
            "task_type": "Base",
            "language": "Chinese",
            "response_format": "pcm",
            "stream": True,
            "stream_format": "audio",
            "ref_audio": ref[0],
            "ref_text": ref[1],
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
        voice = self._resolve_voice(voice_id)
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
        key = (self._resolve_voice(voice_id), text.strip())
        pinned = self._backchannel_cache.get(key)
        return pinned if pinned is not None else self._cache.get(key)

    async def _prefetch(self, text: str, voice: str) -> bytes:
        voice = self._resolve_voice(voice)
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

    async def _warm_voice(self, voice: str) -> None:
        voice = self._resolve_voice(voice)
        if voice in self._warmed:
            return
        lock = self._warm_locks.setdefault(voice, asyncio.Lock())
        async with lock:
            if voice in self._warmed:
                return
            # Pre-synthesize the fixed app greeting; this is silent and warms
            # the talker/codec path so the first real greeting is cache-hit fast.
            try:
                pcm = await self._prefetch(GREETING, voice)
                if pcm:
                    # Pin the greeting so the 8-entry LRU for real replies
                    # cannot evict it after a few turns.
                    self._backchannel_cache[(voice, GREETING)] = pcm
            except Exception as exc:
                logger.warning("TTS greeting prewarm failed voice=%s: %s", voice, exc)
            for phrase in self._backchannel_phrases:
                if (voice, phrase) in self._backchannel_cache:
                    continue  # hand-picked clip
                try:
                    pcm = trim_silence_pcm16(await self._prefetch(phrase, voice))
                    if pcm:
                        self._backchannel_cache[(voice, phrase)] = pcm
                except Exception as exc:
                    logger.warning(
                        "TTS backchannel prewarm failed voice=%s phrase=%r: %s",
                        voice,
                        phrase,
                        exc,
                    )
            self._warmed.add(voice)

    def _schedule_background_warm(self) -> None:
        for spec in self._registry.voices:
            if spec.id in self._warmed:
                continue
            task = asyncio.create_task(
                self._warm_voice(spec.id), name=f"tts-warm-{spec.id}"
            )
            self._warm_tasks.add(task)
            task.add_done_callback(self._warm_tasks.discard)

    async def warmup(self) -> None:
        try:
            await self._client.get("/health")
        except Exception:
            pass
        await self._warm_voice(self._voice_id)
        self._schedule_background_warm()

    async def ensure_warm(self, voice_id: str) -> None:
        await self._warm_voice(voice_id)

    async def close(self) -> None:
        tasks = list(self._warm_tasks)
        self._warm_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._client.aclose()


def make_tts(
    settings: Settings, registry: VoiceRegistry | None = None
) -> TTSBackend:
    registry = registry or load_voice_registry(settings)
    if settings.tts_backend == "vllm":
        return VLLMTTSClient(settings, registry)
    if settings.tts_backend == "qwen3":
        return Qwen3TTSClient(settings, registry)
    return MockTTS(settings, registry)
