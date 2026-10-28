"""Qwen3-TTS worker (separate venv/process).

Run with the TTS venv:
  PYTHONPATH=src python -m echobridge.tts_worker

The worker loads Qwen3-TTS-12Hz-0.6B-Base, builds/caches one clone prompt per
registered voice, and returns 24 kHz PCM16 for one short sentence per request.
The gateway does sentence-level streaming and 20 ms pacing.

Prompt files:
  * default voice: ``prompt.pt`` / ``prompt.meta.json`` (legacy compatible);
  * other voices: ``prompt.<voice_id>.pt`` / ``prompt.<voice_id>.meta.json``.
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
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .config import Settings
from .voices import VoiceRegistry, VoiceSpec, load_voice_registry

MODEL_PATH = os.environ.get(
    "ECHOBRIDGE_TTS_MODEL",
    str(Path.home() / ".cache" / "echobridge" / "models" / "Qwen3-TTS-12Hz-0.6B-Base"),
)
VOICE_DIR = Path(
    os.path.expanduser(
        os.environ.get(
            "ECHOBRIDGE_TTS_VOICE_DIR",
            str(Path.home() / ".cache" / "echobridge" / "voices" / "mingyue"),
        )
    )
)
REFERENCE = Path(
    os.path.expanduser(
        os.environ.get("ECHOBRIDGE_TTS_REFERENCE", str(VOICE_DIR / "reference.wav"))
    )
)
REFERENCE_TEXT = Path(
    os.path.expanduser(
        os.environ.get(
            "ECHOBRIDGE_TTS_REFERENCE_TEXT", str(VOICE_DIR / "reference.txt")
        )
    )
)
PROMPT_PATH = Path(
    os.path.expanduser(
        os.environ.get("ECHOBRIDGE_TTS_PROMPT", str(VOICE_DIR / "prompt.pt"))
    )
)
PROMPT_META = Path(
    os.path.expanduser(
        os.environ.get(
            "ECHOBRIDGE_TTS_PROMPT_META", str(VOICE_DIR / "prompt.meta.json")
        )
    )
)
LANGUAGE = os.environ.get("ECHOBRIDGE_TTS_LANGUAGE", "Chinese")
MAX_NEW_TOKENS = int(os.environ.get("ECHOBRIDGE_TTS_MAX_NEW_TOKENS", "512"))
HOST = os.environ.get("ECHOBRIDGE_TTS_HOST", "127.0.0.1")
PORT = int(os.environ.get("ECHOBRIDGE_TTS_PORT", "18802"))

_registry_settings = Settings(
    tts_voice_dir=str(VOICE_DIR),
    tts_voice_registry=os.environ.get("ECHOBRIDGE_TTS_VOICE_REGISTRY", "").strip(),
    tts_voice_id=os.environ.get("ECHOBRIDGE_TTS_VOICE_ID", "mingyue_v1").strip()
    or "mingyue_v1",
    tts_ref_audio=str(REFERENCE),
    tts_ref_text=str(REFERENCE_TEXT),
)
REGISTRY: VoiceRegistry = load_voice_registry(_registry_settings)

_model = None
_prompts: dict[str, Any] = {}
_prompt_cache_hit: dict[str, bool] = {}
_load_error = ""
_load_s = 0.0
_lock = asyncio.Lock()


def _prompt_paths(voice_id: str) -> tuple[Path, Path]:
    if voice_id == REGISTRY.default_id:
        return PROMPT_PATH, PROMPT_META
    return (
        VOICE_DIR / f"prompt.{voice_id}.pt",
        VOICE_DIR / f"prompt.{voice_id}.meta.json",
    )


def _ref_text_bytes(spec: VoiceSpec) -> bytes:
    raw = spec.ref_text.encode("utf-8")
    # Keep the pre-registry hash for the default voice so an existing
    # prompt.pt remains a cache hit (reference.wav is usually a byte copy of
    # candidate_3.wav).  The hash still includes the audio and model.
    if spec.id == REGISTRY.default_id:
        try:
            legacy = REFERENCE_TEXT.read_bytes()
            if legacy.decode("utf-8", "replace").strip() == spec.ref_text:
                raw = legacy
        except OSError:
            pass
    return raw


def _ref_hash(spec: VoiceSpec) -> str:
    digest = hashlib.sha256()
    digest.update(Path(spec.ref_audio).read_bytes())
    digest.update(b"\0")
    digest.update(_ref_text_bytes(spec))
    digest.update(b"\0")
    digest.update(MODEL_PATH.encode("utf-8"))
    return digest.hexdigest()


def _load_prompt(model, voice_id: str) -> None:
    spec, _ = REGISTRY.resolve(voice_id)
    if spec.id in _prompts:
        return
    prompt_path, meta_path = _prompt_paths(spec.id)
    digest = _ref_hash(spec)
    if prompt_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") == digest:
                loaded = torch.load(
                    prompt_path, map_location="cpu", weights_only=False
                )
                if loaded is None:
                    raise RuntimeError("empty prompt")
                _prompts[spec.id] = loaded
                _prompt_cache_hit[spec.id] = True
                print(f"tts clone prompt cache hit voice={spec.id}", flush=True)
                return
        except Exception as exc:
            print(
                f"tts prompt cache read failed voice={spec.id}: {exc}", flush=True
            )
    if not spec.ref_text:
        raise RuntimeError(f"reference text is empty for voice {spec.id}")
    prompt = model.create_voice_clone_prompt(
        ref_audio=str(spec.ref_audio),
        ref_text=spec.ref_text,
        x_vector_only_mode=False,
    )
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prompt, prompt_path)
    meta_path.write_text(
        json.dumps(
            {"hash": digest, "model": MODEL_PATH, "voice": spec.id},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _prompts[spec.id] = prompt
    _prompt_cache_hit[spec.id] = False
    print(f"tts clone prompt built voice={spec.id}", flush=True)


def _load_model() -> None:
    global _model, _load_error, _load_s
    from qwen_tts import Qwen3TTSModel

    started = time.perf_counter()
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    _load_prompt(_model, REGISTRY.default_id)
    _load_s = time.perf_counter() - started
    _load_error = ""
    # Warmup: generate a short original phrase, then discard it.  No playback.
    _synthesize("你好呀。", 0, REGISTRY.default_id)
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


def _synthesize(text: str, generation: int, voice_id: str) -> tuple[bytes, float, int]:
    spec, _ = REGISTRY.resolve(voice_id)
    if _model is None:
        raise RuntimeError(_load_error or "model not loaded")
    _load_prompt(_model, spec.id)
    prompt = _prompts.get(spec.id)
    if prompt is None:
        raise RuntimeError(f"voice prompt not loaded: {spec.id}")
    started = time.perf_counter()
    wavs, sr = _model.generate_voice_clone(
        text=text,
        language=LANGUAGE,
        voice_clone_prompt=prompt,
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
    default_spec, _ = REGISTRY.resolve(None)
    ref_s = 0.0
    if default_spec.ref_audio.exists():
        try:
            with wave.open(str(default_spec.ref_audio), "rb") as handle:
                ref_s = handle.getnframes() / handle.getframerate()
        except Exception:
            pass
    return JSONResponse(
        {
            "ok": _model is not None and REGISTRY.default_id in _prompts,
            "model": MODEL_PATH,
            "loaded": _model is not None,
            "load_error": _load_error,
            "load_s": round(_load_s, 2),
            "reference": str(default_spec.ref_audio),
            "reference_s": round(ref_s, 2),
            "prompt_cached": _prompt_cache_hit.get(REGISTRY.default_id, False),
            "language": LANGUAGE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "default_voice": REGISTRY.default_id,
            "voices": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "default": spec.id == REGISTRY.default_id,
                    "prompt_cached": spec.id in _prompts,
                }
                for spec in REGISTRY.voices
            ],
            "vram_allocated_mb": round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            if torch.cuda.is_available()
            else 0,
        }
    )


@app.post("/v1/synthesize")
async def synthesize(request: Request) -> Response:
    if _model is None:
        raise HTTPException(status_code=503, detail=_load_error or "model not loaded")
    payload = await request.json()
    text = str(payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text required")
    if len(text) > 200:
        raise HTTPException(status_code=400, detail="text too long")
    generation = int(payload.get("generation") or 0)
    spec, _ = REGISTRY.resolve(payload.get("voice_id"))
    async with _lock:
        try:
            pcm, latency_ms, sample_rate = await asyncio.to_thread(
                _synthesize, text, generation, spec.id
            )
        except Exception as exc:
            print(
                f"tts inference failed voice={spec.id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise HTTPException(status_code=500, detail="tts inference failed") from exc
    duration_s = len(pcm) / 2 / sample_rate if sample_rate else 0.0
    print(
        f"tts request voice={spec.id} chars={len(text)} latency_ms={latency_ms:.1f} "
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
            "X-Voice-Id": spec.id,
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
