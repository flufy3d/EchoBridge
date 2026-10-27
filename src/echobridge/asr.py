"""ASR adapter.

Phase 2 talks to a separate Qwen3-ASR worker over localhost HTTP.  Keeping the
model in its own venv/process avoids the qwen-asr vs qwen-tts dependency clash
and lets the worker stay hot across device sessions.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import Settings
from .logging import get_logger

logger = get_logger("echobridge.asr")

# Qwen3-ASR's context is a prompt, not a constrained decoder.  On noise or
# very short audio it can echo the whole prompt.  Only suppress the obvious
# failure mode: every configured hotword appears and almost nothing else.
_ECHO_PREFIXES = ("technicalterms", "hotwords", "terms", "专有名词", "热词")


def encode_asr_context(context: str) -> str:
    """Encode a UTF-8 context string for the X-ASR-Context HTTP header."""
    return base64.b64encode(context.encode("utf-8")).decode("ascii")


def _normalize_echo(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.casefold())


def is_hotword_echo(text: str, hotwords: tuple[str, ...]) -> bool:
    """Return True when ASR returned the hotword prompt instead of speech."""
    terms = tuple(term.strip() for term in hotwords if term.strip())
    if len(terms) < 3:
        return False
    remainder = _normalize_echo(text)
    if not remainder:
        return False
    hits = 0
    for term in terms:
        token = _normalize_echo(term)
        if token and token in remainder:
            hits += 1
            remainder = remainder.replace(token, "", 1)
    if hits != len(terms):
        return False
    for prefix in _ECHO_PREFIXES:
        remainder = remainder.replace(prefix, "")
    return len(remainder) <= 2


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
    def __init__(
        self, base_url: str, timeout_s: float = 8.0, context: str = ""
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=2.0),
        )
        self._context = context.strip()

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> ASRResult:
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sample_rate),
        }
        if self._context:
            headers["X-ASR-Context"] = encode_asr_context(self._context)
        response = await self._client.post(
            "/v1/transcribe",
            content=pcm,
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text") or "").strip()
        language = str(payload.get("language") or "")
        latency = float(payload.get("latency_ms") or 0.0)
        logger.info(
            "asr result chars=%d language=%s worker_ms=%.1f context=%s",
            len(text),
            language or "-",
            latency,
            "1" if self._context else "0",
        )
        return ASRResult(text, language, latency)

    async def warmup(self) -> None:
        """Open the HTTP connection before the first user turn."""
        await self._client.get("/healthz")

    async def close(self) -> None:
        await self._client.aclose()


def make_asr(settings: Settings, mock_text: str | None = None) -> ASRBackend:
    if settings.asr_backend == "qwen3":
        return Qwen3ASRClient(
            settings.asr_url, settings.asr_timeout_s, settings.asr_context
        )
    return MockASR(mock_text or settings.mock_asr_text)
