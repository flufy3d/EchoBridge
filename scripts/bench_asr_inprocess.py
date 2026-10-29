#!/usr/bin/env python3
"""Direct in-process Qwen3-ASR latency benchmark (no VAD/HTTP/playback).

This measures the model floor: the same trimmed WAV segments used by
``bench_gateway.py`` are fed straight to ``InProcessQwen3ASR``.  Use it to
separate model time from endpoint/speculation effects.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from echobridge.asr import make_asr  # noqa: E402
from echobridge.config import Settings  # noqa: E402
from scripts.bench_gateway import limit_speech, read_wav  # noqa: E402

DEFAULT_WAVS = [
    "var/phase5_question_16k.wav",
    "var/phase5_turn.wav",
    "var/phase5_turn3.wav",
    "var/phase5_turn4.wav",
]


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = max(0, int(round(p / 100.0 * len(ordered))) - 1)
    return ordered[index]


async def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    if settings.asr_backend != "inprocess":
        settings = dataclasses.replace(settings, asr_backend="inprocess")
    asr = make_asr(settings)
    rows = []
    try:
        started = time.perf_counter()
        await asr.warmup()
        warm_s = time.perf_counter() - started
        print(f"warmup_s {warm_s:.2f}", flush=True)
        for item in args.wav or DEFAULT_WAVS:
            path = Path(item)
            if not path.is_absolute():
                path = ROOT / path
            pcm, _ = read_wav(path)
            pcm = limit_speech(pcm, args.max_speech_ms)
            values = []
            text = ""
            for _ in range(args.repeat):
                started = time.perf_counter()
                result = await asr.transcribe(pcm)
                values.append((time.perf_counter() - started) * 1000.0)
                text = result.text
            row = {
                "wav": path.name,
                "audio_ms": len(pcm) // 32,
                "chars": len(text),
                "latency_ms_p50": round(statistics.median(values), 1),
                "latency_ms_max": round(max(values), 1),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        await asr.close()
    if rows:
        all_values = [row["latency_ms_p50"] for row in rows]
        summary = {
            "warmup_s": round(warm_s, 2),
            "latency_ms_p50": round(statistics.median(all_values), 1),
            "latency_ms_max": max(row["latency_ms_max"] for row in rows),
            "rows": rows,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.out:
            Path(args.out).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", action="append", default=[])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-speech-ms", type=int, default=4000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
