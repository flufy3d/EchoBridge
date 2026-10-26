"""Qwen3-TTS worker (separate venv/process).

Run with the TTS venv:
  PYTHONPATH=src python -m echobridge.tts_worker

The worker loads Qwen3-TTS-12Hz-0.6B-Base, builds/caches a clone prompt from the
original reference voice, and returns 24 kHz PCM16 for one short sentence per
request.  The gateway does sentence-level streaming and 20 ms pacing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

MODEL_PATH = os.environ.get(
    "ECHOBRIDGE_TTS_MODEL",
    str(Path.home() / ".cache" / "echobridge" / "models" / "Qwen3-TTS-12Hz-0.6B-Base"),
)
VOICE_DIR = Path(
    os.environ.get(
        "ECHOBRIDGE_TTS_VOICE_DIR",
        str(Path.home() / ".cache" / "echobridge" / "voices" / "mingyue"),
    )
)
REFERENCE = Path(os.environ.get("ECHOBRIDGE_TTS_REFERENCE", str(VOICE_DIR / "reference.wav")))
REFERENCE_TEXT = Path(
    os.environ.get("ECHOBRIDGE_TTS_REFERENCE_TEXT", str(VOICE_DIR / "reference.txt"))
)
PROMPT_PATH = Path(os.environ.get("ECHOBRIDGE_TTS_PROMPT", str(VOICE_DIR / "prompt.pt")))
PROMPT_META = Path(os.environ.get("ECHOBRIDGE_TTS_PROMPT_META", str(VOICE_DIR / "prompt.meta.json")))
LANGUAGE = os.environ.get("ECHOBRIDGE_TTS_LANGUAGE", "Chinese")
MAX_NEW_TOKENS = int(os.environ.get("ECHOBRIDGE_TTS_MAX_NEW_TOKENS", "512"))
HOST = os.environ.get("ECHOBRIDGE_TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("ECHOBRIDGE_TTS_PORT", "18802"))

_model = None
_prompt = None
_load_error = ""
_load_s = 0.0
_prompt_cached = False
_lock = asyncio.Lock()


def _ref_hash() -> str:
    digest = hashlib.sha256()
    digest.update(REFERENCE.read_bytes())
    digest.update(b"\0")
    digest.update(REFERENCE_TEXT.read_bytes())
    digest.update(b"\0")
    digest.update(MODEL_PATH.encode("utf-8"))
    return digest.hexdigest()


def _load_prompt(model):
    global _prompt, _prompt_cached
    digest = _ref_hash()
    if PROMPT_PATH.exists() and PROMPT_META.exists():
        try:
            meta = json.loads(PROMPT_META.read_text(encoding="utf-8"))
            if meta.get("hash") == digest:
                _prompt = torch.load(PROMPT_PATH, map_location="cpu", weights_only=False)
                _prompt_cached = True
                print("tts clone prompt cache hit", flush=True)
                return
        except Exception as exc:
            print(f"tts prompt cache read failed: {exc}", flush=True)
    ref_text = REFERENCE_TEXT.read_text(encoding="utf-8").strip()
    if not ref_text:
        raise RuntimeError("reference.txt is empty")
    _prompt = model.create_voice_clone_prompt(
        ref_audio=str(REFERENCE),
        ref_text=ref_text,
        x_vector_only_mode=False,
    )
    PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_prompt, PROMPT_PATH)
    PROMPT_META.write_text(
        json.dumps({"hash": digest, "model": MODEL_PATH}, ensure_ascii=False),
        encoding="utf-8",
    )
    _prompt_cached = False
    print("tts clone prompt built", flush=True)


def _load_model():
    global _model, _load_error, _load_s
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    _load_prompt(_model)
    _load_s = time.perf_counter() - started
    _load_error = ""
    # Warmup: generate a short original phrase, then discard it.  No playback.
    _synthesize("你好呀。", 0)
    print(
        f"tts model loaded path={MODEL_PATH} load_s={_load_s:.2f} "
        f"vram_mb={torch.cuda.memory_allocated() / 1024 / 1024:.1f}",
        flush=True,
    )


def _to_pcm16(wav: np.ndarray) -> bytes:
    audio = np.asarray(wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def _synthesize(text: str, generation: int) -> tuple[bytes, float, int]:
    if _model is None or _prompt is None:
        raise RuntimeError(_load_error or "model/prompt not loaded")
    started = time.perf_counter()
    wavs, sr = _model.generate_voice_clone(
        text=text,
        language=LANGUAGE,
        voice_clone_prompt=_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not wavs:
        raise RuntimeError("tts returned no audio")
    pcm = _to_pcm16(wavs[0])
    return pcm, elapsed_ms, int(sr)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _load_error
    try:
        await asyncio.to_thread(_load_model)
    except Exception as exc:
        _load_error = f"{type(exc).__name__}: {exc}"
        print(f"tts model load failed: {_load_error}", flush=True)
    yield


app = FastAPI(title="EchoBridge TTS Worker", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    ref_s = 0.0
    if REFERENCE.exists():
        try:
            with wave.open(str(REFERENCE), "rb") as handle:
                ref_s = handle.getnframes() / handle.getframerate()
        except Exception:
            pass
    return JSONResponse(
        {
            "ok": _model is not None and _prompt is not None,
            "model": MODEL_PATH,
            "loaded": _model is not None,
            "load_error": _load_error,
            "load_s": round(_load_s, 2),
            "reference": str(REFERENCE),
            "reference_s": round(ref_s, 2),
            "prompt_cached": _prompt_cached,
            "language": LANGUAGE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else 0,
        }
    )


@app.post("/v1/synthesize")
async def synthesize(request: Request) -> Response:
    if _model is None or _prompt is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    payload = await request.json()
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if len(text) > 200:
        raise HTTPException(status_code=400, detail="text too long")
    generation = int(payload.get("generation") or 0)
    async with _lock:
        try:
            pcm, latency_ms, sample_rate = await asyncio.to_thread(
                _synthesize, text, generation
            )
        except Exception as exc:
            print(f"tts inference failed: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(status_code=500, detail="tts inference failed") from exc
    duration_s = len(pcm) / 2 / sample_rate if sample_rate else 0.0
    print(
        f"tts request chars={len(text)} latency_ms={latency_ms:.1f} "
        f"audio_s={duration_s:.2f} rtf={latency_ms / max(1.0, duration_s * 1000):.2f}",
        flush=True,
    )
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(sample_rate),
            "X-Latency-Ms": f"{latency_ms:.1f}",
            "X-Generation": str(generation),
        },
    )


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
