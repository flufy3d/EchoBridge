"""Qwen3-ASR worker (separate venv/process).

Run with the ASR venv:
  PYTHONPATH=src python -m echobridge.asr_worker

The gateway only sees a localhost HTTP endpoint; model loading, BF16, CUDA and
transformers stay in this process so the qwen-asr/qwen-tts dependency trees do
not collide.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MODEL_PATH = os.environ.get(
    "ECHOBRIDGE_ASR_MODEL",
    str(Path.home() / ".cache" / "echobridge" / "models" / "Qwen3-ASR-0.6B"),
)
LANGUAGE = os.environ.get("ECHOBRIDGE_ASR_LANGUAGE", "").strip() or None
MAX_NEW_TOKENS = int(os.environ.get("ECHOBRIDGE_ASR_MAX_NEW_TOKENS", "192"))
HOST = os.environ.get("ECHOBRIDGE_ASR_HOST", "127.0.0.1")
PORT = int(os.environ.get("ECHOBRIDGE_ASR_PORT", "18801"))

_model = None
_load_error = ""
_lock = asyncio.Lock()
_loaded_at = 0.0


def _load_model():
    global _model, _load_error, _loaded_at
    if _model is not None:
        return
    from qwen_asr import Qwen3ASRModel

    started = time.perf_counter()
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=MAX_NEW_TOKENS,
        # FlashAttention-2 is optional; SDPA is the portable default.
        # attn_implementation="flash_attention_2",
    )
    _loaded_at = time.perf_counter() - started
    _load_error = ""
    # Warm up kernels and the audio frontend.  This is not user audio.
    warm = np.zeros(16000, dtype=np.float32)
    _model.transcribe(audio=(warm, 16000), language=LANGUAGE)
    torch.cuda.synchronize()
    print(
        f"asr model loaded path={MODEL_PATH} load_s={_loaded_at:.2f} "
        f"vram_mb={torch.cuda.memory_allocated() / 1024 / 1024:.1f}",
        flush=True,
    )


def _transcribe(pcm: bytes, sample_rate: int) -> dict:
    if _model is None:
        raise RuntimeError(_load_error or "model not loaded")
    if sample_rate != 16000:
        raise ValueError("only 16 kHz input is supported")
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    started = time.perf_counter()
    results = _model.transcribe(audio=(audio, sample_rate), language=LANGUAGE)
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


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _load_error
    try:
        await asyncio.to_thread(_load_model)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        print(f"asr model load failed: {_load_error}", flush=True)
    yield


app = FastAPI(title="EchoBridge ASR Worker", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "ok": _model is not None,
            "model": MODEL_PATH,
            "loaded": _model is not None,
            "load_error": _load_error,
            "load_s": round(_loaded_at, 2),
            "language": LANGUAGE or "auto",
            "max_new_tokens": MAX_NEW_TOKENS,
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else 0,
        }
    )


@app.post("/v1/transcribe")
async def transcribe(request: Request) -> JSONResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    sample_rate = int(request.headers.get("x-sample-rate", "16000"))
    body = await request.body()
    if not body or len(body) % 2:
        raise HTTPException(status_code=400, detail="PCM16 body required")
    async with _lock:
        try:
            result = await asyncio.to_thread(_transcribe, body, sample_rate)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            print(f"asr inference failed: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(status_code=500, detail="asr inference failed") from exc
    # Never log the transcript itself.
    print(
        f"asr request bytes={len(body)} latency_ms={result['latency_ms']:.1f} "
        f"chars={len(result['text'])}",
        flush=True,
    )
    return JSONResponse(result)


def main() -> None:
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("ECHOBRIDGE_LOG_LEVEL", "INFO").lower(),
        access_log=False,
    )


if __name__ == "__main__":
    main()
