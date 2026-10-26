"""Phase 1 mock TTS.

The real Qwen3-TTS worker starts in Phase 3.  For Phase 1 we synthesize a
deterministic tone sequence (or replay a local WAV) so the transport, pacing and
cancel behavior can be verified on the real device.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from .audio import load_wav_mono_24k, silence_pcm16, tone_pcm16
from .config import Settings


class TTSBackend(Protocol):
    def synthesize(
        self, text: str, voice_id: str, generation: int
    ) -> AsyncIterator[bytes]: ...


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
        # Feed in 100 ms blocks; the pacer turns them into 20 ms frames.
        block = 4_800
        for offset in range(0, len(pcm), block):
            chunk = pcm[offset : offset + block]
            if chunk:
                yield chunk
