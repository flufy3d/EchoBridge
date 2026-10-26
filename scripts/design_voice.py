#!/usr/bin/env python3
"""Offline Qwen3-TTS VoiceDesign reference generation.

Generates 3 candidate original voice references, writes them outside the repo,
and selects a candidate by simple objective checks (5-9 s, no clipping).  It does
not play audio.  The user can audition the candidates later and replace
reference.wav if desired.
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

DEFAULT_MODEL = str(
    Path.home()
    / ".cache"
    / "echobridge"
    / "models"
    / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
DEFAULT_DIR = Path.home() / ".cache" / "echobridge" / "voices" / "mingyue"
DEFAULT_TEXT = "你好呀，我是溟月。今天也想陪你聊聊天，可以吗？"
DEFAULT_INSTRUCT = (
    "年轻女性，二次元可爱声线，明亮、软糯、略带俏皮，语速中等偏快，"
    "句尾轻微上扬；自然清晰，不要过度夹子音/萝莉音，不要播音腔。"
)


def wav_stats(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        return {"duration_s": 0.0, "peak": 1.0, "rms": 0.0}
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    return {
        "duration_s": round(len(audio) / rate, 2),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("ECHOBRIDGE_VOICE_DESIGN_MODEL", DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--instruct", default=DEFAULT_INSTRUCT)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "reference.wav").exists() and not args.force:
        print(f"reference already exists: {out / 'reference.wav'} (use --force)")
        return
    if not Path(args.model).exists():
        raise SystemExit(f"VoiceDesign model not found: {args.model}")

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    rows = []
    for index in range(1, args.candidates + 1):
        wavs, sr = model.generate_voice_design(
            text=args.text,
            instruct=args.instruct,
            language="Chinese",
        )
        audio = np.asarray(wavs[0], dtype=np.float32)
        path = out / f"candidate_{index}.wav"
        sf.write(str(path), audio, sr)
        stats = wav_stats(path)
        stats.update({"candidate": index, "path": str(path), "sr": sr})
        rows.append(stats)
        print(json.dumps(stats, ensure_ascii=False), flush=True)

    # Prefer 5-9 s, no clipping, healthy RMS; fall back to closest duration.
    def score(row: dict) -> tuple:
        duration = row["duration_s"]
        in_range = 5.0 <= duration <= 9.0
        return (
            0 if in_range else 1,
            0 if row["peak"] < 0.99 else 1,
            abs(duration - 7.0),
            -row["rms"],
        )

    best = sorted(rows, key=score)[0]
    best_path = Path(best["path"])
    (out / "reference.wav").write_bytes(best_path.read_bytes())
    (out / "reference.txt").write_text(args.text, encoding="utf-8")
    meta = {
        "selected": best["candidate"],
        "text": args.text,
        "instruct": args.instruct,
        "model": args.model,
        "candidates": rows,
        "stats": best,
    }
    (out / "voice.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Remove a stale clone prompt if the reference changed.
    for name in ("prompt.pt", "prompt.meta.json"):
        p = out / name
        if p.exists():
            p.unlink()
    print(f"selected candidate {best['candidate']} -> {out / 'reference.wav'}")


if __name__ == "__main__":
    main()
