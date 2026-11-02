"""ASR adapter.

Phase 2 talks to a separate Qwen3-ASR worker over localhost HTTP.  Keeping the
model in its own venv/process avoids the qwen-asr vs qwen-tts dependency clash
and lets the worker stay hot across device sessions.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import os
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from .config import Settings
from .gpu import (
    PRIORITY_ASR_FINAL,
    PRIORITY_ASR_SPECULATIVE,
    get_gpu_scheduler,
)
from .logging import get_logger

logger = get_logger("echobridge.asr")

# Qwen3-ASR's context is a prompt, not a constrained decoder.  On noise or
# very short audio it can echo the prompt, often only part of it.  Suppress
# results that are just 3+ configured hotwords (plus a little punctuation).
_ECHO_PREFIXES = ("technicalterms", "hotwords", "terms", "专有名词", "热词")


def encode_asr_context(context: str) -> str:
    """Encode a UTF-8 context string for the X-ASR-Context HTTP header."""
    return base64.b64encode(context.encode("utf-8")).decode("ascii")


def parse_hotwords(raw: Any) -> tuple[str, ...] | None:
    """Parse an ASR hotword list from a JSON array or a delimited string."""
    if isinstance(raw, list):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    if isinstance(raw, str):
        return tuple(
            part.strip()
            for part in re.split(r"[|,，、;；\n]+", raw)
            if part.strip()
        )
    return None


def _normalize_echo(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.casefold())


def is_hotword_echo(text: str, hotwords: tuple[str, ...]) -> bool:
    """Return True when ASR returned the hotword prompt instead of speech.

    Qwen3-ASR's ``context`` is a free-form prompt, not a constrained decoder.
    On noise or very short audio it often answers with *part* of the configured
    term list (typically the Chinese terms, in prompt order) rather than a
    transcript.  Requiring the whole list -- or an exact match for every term,
    since ASR can mangle one -- misses that case, so treat a result that is 3+
    hotwords and almost nothing else as an echo.
    """
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
    if hits < 3:
        return False
    for prefix in _ECHO_PREFIXES:
        remainder = remainder.replace(prefix, "")
    # Leftover budget covers punctuation plus one short misrecognized term;
    # "甲，乙，丙，你们好" keeps three characters and is real speech.
    return len(remainder) <= 2


@dataclass(slots=True)
class ASRResult:
    text: str
    language: str = ""
    latency_ms: float = 0.0


# 0.5 s of 16 kHz mono PCM silence.  The vLLM ASR path accepts it and returns
# an empty transcript, which is enough to exercise the CUDA graph and the HTTP
# stack without sending or logging user audio.
SILENCE_PCM = b"\x00\x00" * 8_000


class ASRUnavailableError(RuntimeError):
    """The ASR worker is down, restarting, or temporarily overloaded."""


class ASRError(RuntimeError):
    """The ASR worker rejected this specific request (non-retryable)."""


class AsrHealth:
    """Process-local availability state shared by the ASR adapters.

    This is deliberately a status record, not a circuit breaker: a session
    still attempts the request after a failure, so a worker restarted by
    systemd recovers on the next utterance without a separate probe.  The
    fields are only used for classification, readiness and metrics.
    """

    def __init__(self) -> None:
        self.available = True
        self.last_ok = 0.0
        self.last_error = ""
        self.last_error_at = 0.0
        self.unavailable_total = 0
        self.error_total = 0
        self.retry_total = 0
        self.retry_success_total = 0

    def mark_ok(self) -> None:
        self.available = True
        self.last_ok = time.time()
        self.last_error = ""

    def mark_unavailable(self, reason: str) -> None:
        self.available = False
        self.last_error = reason[:120]
        self.last_error_at = time.time()
        self.unavailable_total += 1

    def mark_error(self, reason: str) -> None:
        # A 4xx/parse error means the worker is reachable; do not mask it as
        # unavailable, but keep the last error for diagnostics.
        self.last_error = reason[:120]
        self.last_error_at = time.time()
        self.error_total += 1

    def record_retry(self) -> None:
        self.retry_total += 1

    def record_retry_success(self) -> None:
        self.retry_success_total += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "available": self.available,
            "last_ok": self.last_ok,
            "last_error": self.last_error,
            "unavailable_total": self.unavailable_total,
            "error_total": self.error_total,
            "retry_total": self.retry_total,
            "retry_success_total": self.retry_success_total,
        }

    def metrics_lines(self) -> list[str]:
        return [
            "# HELP echobridge_asr_available ASR worker is currently usable.",
            "# TYPE echobridge_asr_available gauge",
            f"echobridge_asr_available {1 if self.available else 0}",
            "# HELP echobridge_asr_unavailable_total "
            "ASR attempts that failed because the worker was unavailable.",
            "# TYPE echobridge_asr_unavailable_total counter",
            f"echobridge_asr_unavailable_total {self.unavailable_total}",
            "# HELP echobridge_asr_error_total "
            "ASR attempts rejected with a non-retryable error.",
            "# TYPE echobridge_asr_error_total counter",
            f"echobridge_asr_error_total {self.error_total}",
            "# HELP echobridge_asr_retry_total "
            "ASR retry attempts triggered by unavailability.",
            "# TYPE echobridge_asr_retry_total counter",
            f"echobridge_asr_retry_total {self.retry_total}",
            "# HELP echobridge_asr_retry_success_total "
            "ASR retries that recovered the same utterance.",
            "# TYPE echobridge_asr_retry_success_total counter",
            f"echobridge_asr_retry_success_total {self.retry_success_total}",
            "# HELP echobridge_asr_last_ok_timestamp_seconds "
            "Unix time of the last successful ASR request.",
            "# TYPE echobridge_asr_last_ok_timestamp_seconds gauge",
            f"echobridge_asr_last_ok_timestamp_seconds {self.last_ok:.3f}",
        ]


class ASRBackend(Protocol):
    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        context: str | None = None,
        *,
        speculative: bool = False,
    ) -> ASRResult: ...

    async def warmup(self) -> None: ...

    async def close(self) -> None: ...


class MockASR:
    def __init__(self, text: str) -> None:
        self._text = text.strip()
        self.asr_health = AsrHealth()

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        context: str | None = None,
        *,
        speculative: bool = False,
    ) -> ASRResult:
        self.asr_health.mark_ok()
        return ASRResult(self._text, "zh", 0.0)

    async def warmup(self) -> None:
        self.asr_health.mark_ok()

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
        self.asr_health = AsrHealth()

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        context: str | None = None,
        *,
        speculative: bool = False,
    ) -> ASRResult:
        # None keeps the gateway-wide ECHOBRIDGE_ASR_HOTWORDS default; an
        # explicit "" disables it for a session that cleared its hotword list.
        effective_context = self._context if context is None else context.strip()
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Sample-Rate": str(sample_rate),
        }
        if effective_context:
            headers["X-ASR-Context"] = encode_asr_context(effective_context)
        try:
            response = await self._client.post(
                "/v1/transcribe",
                content=pcm,
                headers=headers,
            )
        except httpx.TimeoutException as exc:
            self.asr_health.mark_unavailable("timeout")
            raise ASRUnavailableError("ASR worker timeout") from exc
        except httpx.HTTPError as exc:
            self.asr_health.mark_unavailable(type(exc).__name__)
            raise ASRUnavailableError(
                f"ASR worker {type(exc).__name__}"
            ) from exc
        if response.status_code != 200:
            status = response.status_code
            if status in (408, 429, 500, 502, 503, 504):
                self.asr_health.mark_unavailable(f"HTTP {status}")
                raise ASRUnavailableError(f"ASR worker HTTP {status}")
            self.asr_health.mark_error(f"HTTP {status}")
            raise ASRError(f"ASR worker HTTP {status}")
        try:
            payload = response.json()
        except ValueError as exc:
            self.asr_health.mark_error("invalid JSON")
            raise ASRError("ASR worker returned invalid JSON") from exc
        text = str(payload.get("text") or "").strip()
        language = str(payload.get("language") or "")
        latency = float(payload.get("latency_ms") or 0.0)
        self.asr_health.mark_ok()
        logger.info(
            "asr result chars=%d language=%s worker_ms=%.1f context=%s",
            len(text),
            language or "-",
            latency,
            "1" if effective_context else "0",
        )
        return ASRResult(text, language, latency)

    async def warmup(self) -> None:
        """Open the HTTP connection before the first user turn."""
        await self._client.get("/healthz")

    async def close(self) -> None:
        await self._client.aclose()


class VLLMASRClient:
    """Qwen3-ASR served by vLLM's OpenAI-compatible transcription API.

    vLLM 0.32 runs the decoder with CUDA graphs, which removes the per-token
    GPU launch overhead that dominates the HF path on this CPU.  Measured on
    an RTX 3070: 4 s audio in ~85 ms, 14 s audio in ~450 ms (including HTTP
    and multipart upload).

    The server is deliberately external: loading vLLM into the gateway would
    require the TTS venv and a second CUDA context in the main process.
    """

    _LANGUAGE_CODES = {
        "chinese": "zh",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
        "cantonese": "yue",
    }

    def __init__(self, settings: Settings) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.asr_url.rstrip("/"),
            timeout=httpx.Timeout(settings.asr_timeout_s, connect=2.0),
        )
        self._context = settings.asr_context
        self._model = settings.asr_vllm_model
        self._language = self._LANGUAGE_CODES.get(
            (settings.asr_language or "").strip().lower(),
            (settings.asr_language or "").strip().lower(),
        )
        self._max_tokens = max(1, settings.asr_max_new_tokens)
        self.asr_health = AsrHealth()

    @staticmethod
    def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm)
        return buffer.getvalue()

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        context: str | None = None,
        *,
        speculative: bool = False,
    ) -> ASRResult:
        effective_context = self._context if context is None else context.strip()
        data: dict[str, str | int] = {
            "model": self._model,
            "response_format": "json",
            "temperature": "0",
            "max_completion_tokens": self._max_tokens,
        }
        if self._language:
            data["language"] = self._language
        if effective_context:
            # Qwen3-ASR treats this as a system/context prompt, not as a
            # weighted hotword table.
            data["prompt"] = effective_context

        started = time.perf_counter()
        try:
            response = await self._client.post(
                "/v1/audio/transcriptions",
                files={
                    "file": (
                        "audio.wav",
                        self._wav_bytes(pcm, sample_rate),
                        "audio/wav",
                    )
                },
                data=data,
            )
        except httpx.TimeoutException as exc:
            self.asr_health.mark_unavailable("timeout")
            raise ASRUnavailableError("vLLM ASR timeout") from exc
        except httpx.HTTPError as exc:
            self.asr_health.mark_unavailable(type(exc).__name__)
            raise ASRUnavailableError(
                f"vLLM ASR {type(exc).__name__}"
            ) from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            status = response.status_code
            if status in (408, 429, 500, 502, 503, 504):
                self.asr_health.mark_unavailable(f"HTTP {status}")
                raise ASRUnavailableError(f"vLLM ASR HTTP {status}")
            self.asr_health.mark_error(f"HTTP {status}")
            raise ASRError(f"vLLM ASR HTTP {status}")
        try:
            payload = response.json()
        except ValueError as exc:
            self.asr_health.mark_error("invalid JSON")
            raise ASRError("vLLM ASR returned invalid JSON") from exc
        text = str(payload.get("text") or "").strip()
        self.asr_health.mark_ok()
        logger.info(
            "asr vllm result chars=%d wall_ms=%.1f context=%s",
            len(text),
            wall_ms,
            "1" if effective_context else "0",
        )
        return ASRResult(text, self._language, wall_ms)

    async def warmup(self) -> None:
        # A short silent request warms the CUDA graph path as well as the
        # keep-alive connection; failures are reported by the caller.
        try:
            await self.transcribe(SILENCE_PCM, 16000)
        except Exception:
            await self._client.get("/health")

    async def close(self) -> None:
        await self._client.aclose()


class InProcessQwen3ASR:
    """Qwen3-ASR loaded directly in the gateway process.

    This removes the localhost HTTP round-trip, the PCM body copy and the
    base64 context header.  Inference still runs in a worker thread so the
    asyncio loop (WebSocket receive, pacer, VAD) stays responsive; the shared
    :class:`~echobridge.gpu.GPUScheduler` serialises it against in-process TTS.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model_path = str(Path(os.path.expanduser(settings.asr_model)))
        self._language = settings.asr_language or None
        self._max_new_tokens = max(1, settings.asr_max_new_tokens)
        self._device = settings.asr_device
        self._attn = settings.asr_attn_implementation or "sdpa"
        self._scheduler = get_gpu_scheduler()
        self._model: Any = None
        self._supports_context: bool | None = None
        self._load_error = ""
        self._load_s = 0.0
        self._loaded_at = 0.0
        self._thread_lock = threading.Lock()
        self._load_lock: asyncio.Lock | None = None
        self._context_warned = False
        self.asr_health = AsrHealth()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def backend(self) -> str:
        return "inprocess"

    def _ensure_load_lock(self) -> asyncio.Lock:
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        return self._load_lock

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._ensure_load_lock():
            if self._model is not None:
                return
            try:
                await self._scheduler.run(
                    "asr-load",
                    self._load_sync,
                    priority=PRIORITY_ASR_FINAL,
                    timeout=None,
                )
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                self.asr_health.mark_unavailable(f"load: {type(exc).__name__}")
                logger.exception("inprocess asr load failed")
                raise

    def _load_sync(self) -> None:
        if self._model is not None:
            return
        with self._thread_lock:
            if self._model is not None:
                return
            # Import lazily: mock/energy test runs never touch qwen_asr/torch.
            import numpy as np
            import torch
            from qwen_asr import Qwen3ASRModel

            started = time.perf_counter()
            kwargs: dict[str, Any] = {}
            if self._attn:
                kwargs["attn_implementation"] = self._attn
            model = Qwen3ASRModel.from_pretrained(
                self._model_path,
                dtype=torch.bfloat16,
                device_map=self._device,
                max_inference_batch_size=1,
                max_new_tokens=self._max_new_tokens,
                **kwargs,
            )
            try:
                self._supports_context = (
                    "context" in inspect.signature(model.transcribe).parameters
                )
            except (TypeError, ValueError):
                self._supports_context = False
            if self._settings.asr_compile:
                try:
                    model.thinker.model = torch.compile(
                        model.thinker.model, dynamic=True
                    )
                    logger.info("inprocess asr decoder torch.compile enabled")
                except Exception as exc:
                    logger.warning("inprocess asr torch.compile failed: %s", exc)
            # Warm kernels and the audio frontend with silence.  No playback.
            warm = np.zeros(16000, dtype=np.float32)
            model.transcribe(audio=(warm, 16000), language=self._language)
            torch.cuda.synchronize()
            self._model = model
            self._loaded_at = time.perf_counter()
            self._load_s = self._loaded_at - started
            self._load_error = ""
            self.asr_health.mark_ok()
            logger.info(
                "inprocess asr loaded path=%s load_s=%.2f vram_mb=%.1f context=%s",
                self._model_path,
                self._load_s,
                torch.cuda.memory_allocated() / (1024 * 1024),
                self._supports_context,
            )

    async def transcribe(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        context: str | None = None,
        *,
        speculative: bool = False,
    ) -> ASRResult:
        if sample_rate != 16000:
            raise ValueError("only 16 kHz input is supported")
        if not pcm:
            return ASRResult("", "", 0.0)
        priority = (
            PRIORITY_ASR_SPECULATIVE if speculative else PRIORITY_ASR_FINAL
        )
        effective_context = (context or "").strip()
        started = time.perf_counter()
        try:
            await self._ensure_loaded()
            if self._model is None:
                raise RuntimeError(self._load_error or "ASR model not loaded")
            payload = await self._scheduler.run(
                "asr-spec" if speculative else "asr",
                self._infer_sync,
                pcm,
                context=effective_context,
                priority=priority,
                timeout=self._settings.asr_timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self.loaded:
                # _ensure_loaded already recorded the load failure.
                raise ASRUnavailableError(
                    f"in-process ASR unavailable: {type(exc).__name__}"
                ) from exc
            self.asr_health.mark_error(type(exc).__name__)
            raise ASRError(
                f"in-process ASR error: {type(exc).__name__}"
            ) from exc
        wall_ms = (time.perf_counter() - started) * 1000.0
        self.asr_health.mark_ok()
        logger.info(
            "asr result chars=%d language=%s infer_ms=%.1f wall_ms=%.1f "
            "spec=%s context=%s",
            len(payload["text"]),
            payload["language"] or "-",
            payload["latency_ms"],
            wall_ms,
            "1" if speculative else "0",
            "1" if effective_context else "0",
        )
        return ASRResult(payload["text"], payload["language"], payload["latency_ms"])

    def _infer_sync(self, pcm: bytes, *, context: str) -> dict[str, Any]:
        import numpy as np
        import torch

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        kwargs: dict[str, Any] = {}
        if context and self._supports_context:
            kwargs["context"] = context
        elif context and not self._context_warned:
            logger.warning(
                "inprocess asr context ignored: installed qwen-asr has no "
                "context parameter"
            )
            self._context_warned = True
        started = time.perf_counter()
        results = self._model.transcribe(
            audio=(audio, 16000),
            language=self._language,
            **kwargs,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not results:
            return {"text": "", "language": "", "latency_ms": elapsed_ms}
        result = results[0]
        return {
            "text": str(getattr(result, "text", "") or "").strip(),
            "language": str(getattr(result, "language", "") or ""),
            "latency_ms": elapsed_ms,
        }

    async def warmup(self) -> None:
        await self._ensure_loaded()

    def health(self) -> dict[str, object]:
        memory = self._scheduler.memory() if self._model is not None else None
        return {
            "ok": self._model is not None,
            "backend": "inprocess",
            "model": self._model_path,
            "loaded": self._model is not None,
            "load_s": round(self._load_s, 2),
            "load_error": self._load_error,
            "language": self._language or "auto",
            "max_new_tokens": self._max_new_tokens,
            "supports_context": self._supports_context,
            "vram_allocated_mb": round(memory.allocated_mb, 1) if memory else 0.0,
            "vram_reserved_mb": round(memory.reserved_mb, 1) if memory else 0.0,
            "device": memory.device if memory else "",
        }

    async def close(self) -> None:
        if self._model is None:
            return
        self._model = None
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def make_asr(settings: Settings, mock_text: str | None = None) -> ASRBackend:
    if settings.asr_backend == "inprocess":
        return InProcessQwen3ASR(settings)
    if settings.asr_backend == "vllm":
        return VLLMASRClient(settings)
    if settings.asr_backend == "qwen3":
        return Qwen3ASRClient(
            settings.asr_url, settings.asr_timeout_s, settings.asr_context
        )
    return MockASR(mock_text or settings.mock_asr_text)
