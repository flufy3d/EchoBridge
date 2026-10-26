"""ASR adapter.

Phase 2 talks to a separate Qwen3-ASR worker over localhost HTTP.  Keeping the
model in its own venv/process avoids the qwen-asr vs qwen-tts dependency clash
and lets the worker stay hot across device sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings
from .logging import get_logger

logger = get_logger("echobridge.asr")


@dataclass(slots=True)
class ASRResult:
    text: str
    language: str = ""
    latency_ms: float = 0.0


class ASRBackend(Protocol):
    async def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> ASRResult: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...


class MockASR:
    def __init__(self, text: str) -> None:
        self._text = text.strip()

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> ASRResult:
        return ASRResult(self._text, "zh", 0.0)

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


class Qwen3ASRClient:
    def __init__(self, base_url: str, timeout_s: float = 8.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=2.0),
        )

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> ASRResult:
        response = await self._client.post(
            "/v1/transcribe",
            content=pcm,
            headers={"Content-Type": "application/octet-stream", "X-Sample-Rate": str(sample_rate)},
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text") or "").strip()
        language = str(payload.get("language") or "")
        latency = float(payload.get("latency_ms") or 0.0)
        logger.info("asr result chars=%d language=%s worker_ms=%.1f", len(text), language or "-", latency)
        return ASRResult(text, language, latency)

    async def warmup(self) -> None:
        """Open the HTTP connection before the first user turn."""
        await self._client.get("/healthz")

    async def close(self) -> None:
        await self._client.aclose()


def make_asr(settings: Settings, mock_text: str | None = None) -> ASRBackend:
    if settings.asr_backend == "qwen3":
        return Qwen3ASRClient(settings.asr_url, settings.asr_timeout_s)
    return MockASR(mock_text or settings.mock_asr_text)
