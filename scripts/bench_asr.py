#!/usr/bin/env python3
"""ASR quality/latency benchmark against the local Qwen3-ASR worker.

Uses a public Qwen3-ASR sample with a known transcript.  Noise/far-field cases
are *simulated* until the X10A real recordings are available; they are labelled
accordingly and must not be reported as device WER.

Run with the ASR venv (numpy/scipy available):
  python scripts/bench_asr.py --wav ~/.cache/echobridge/bench/asr_zh.wav \
      --reference "甚至出现交易几乎停滞的情况。" --repeat 3
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import wave
from pathlib import Path

import httpx
import numpy as np


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise SystemExit("WAV must be mono PCM16")
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def normalize_text(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = float(np.mean(audio**2)) or 1e-8
    noise_power = power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(noise_power), audio.shape).astype(np.float32)
    return np.clip(audio + noise, -1.0, 1.0)


def low_pass(audio: np.ndarray, rate: int, cutoff_hz: float) -> np.ndarray:
    try:
        from scipy.signal import butter, sosfilt

        sos = butter(4, cutoff_hz / (rate / 2.0), btype="lowpass", output="sos")
        return sosfilt(sos, audio).astype(np.float32)
    except Exception:
        # Portable fallback: 5-tap moving average.
        kernel = np.ones(5, dtype=np.float32) / 5.0
        return np.convolve(audio, kernel, mode="same").astype(np.float32)


def variants(audio: np.ndarray, rate: int) -> dict[str, np.ndarray]:
    return {
        "clean": audio,
        "tv_white_snr10": add_noise(audio, 10.0),
        "far_lpf4k_snr15": add_noise(low_pass(audio, rate, 4000.0) * 0.25, 15.0),
    }


def post(client: httpx.Client, url: str, audio: np.ndarray, rate: int) -> tuple[dict, float]:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16).tobytes()
    started = time.perf_counter()
    response = client.post(
        url + "/v1/transcribe",
        content=pcm,
        headers={"X-Sample-Rate": str(rate), "Content-Type": "application/octet-stream"},
        timeout=120.0,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    response.raise_for_status()
    return response.json(), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18801")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    audio, rate = load_wav(args.wav)
    duration = len(audio) / rate
    rows = []
    with httpx.Client() as client:
        for name, variant in variants(audio, rate).items():
            times = []
            last = None
            for _ in range(args.repeat):
                payload, wall_ms = post(client, args.url, variant, rate)
                times.append(wall_ms)
                last = payload
            rows.append(
                {
                    "case": name,
                    "duration_s": round(duration, 2),
                    "text": last["text"],
                    "reference": args.reference,
                    "cer": round(cer(args.reference, last["text"]), 4),
                    "latency_ms_p50": round(statistics.median(times), 1),
                    "latency_ms_p95": round(sorted(times)[max(0, math.ceil(0.95 * len(times)) - 1)], 1),
                    "worker_ms": round(float(last.get("latency_ms", 0)), 1),
                }
            )

    print(json.dumps(rows, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
