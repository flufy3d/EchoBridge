#!/usr/bin/env python3
"""Benchmark the in-process Qwen3-TTS backend. No playback.

Compares the same five-sentence set as ``bench_tts_vllm.py`` and reports first
chunk p50/max, RTF p50 and process CUDA-allocator memory.  Use ``nvidia-smi``
alongside it for whole-card usage.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from echobridge.config import Settings  # noqa: E402
from echobridge.tts import make_tts  # noqa: E402

TEXTS = [
    "你好呀，我在呢。",
    "今天也要开开心心的呀。",
    "现在是晚上十一点，该准备休息啦。",
    "我看看哦，这个大概是一个蓝色的杯子。",
    "23 乘以 47 等于 1081。",
]


async def run(args: argparse.Namespace) -> int:
    settings = dataclasses.replace(
        Settings.from_env(), tts_backend=args.backend
    )
    tts = make_tts(settings)
    rows = []
    try:
        started = time.perf_counter()
        await tts.warmup()
        warm_s = time.perf_counter() - started
        print(f"warmup_s {warm_s:.2f}", flush=True)
        for text in TEXTS:
            firsts: list[float] = []
            totals: list[float] = []
            durations: list[float] = []
            for _ in range(args.repeat):
                # Each repeat must actually run the model, otherwise repeats
                # 2/3 are cache hits and p50 becomes meaningless.
                cache = getattr(tts, "_cache", None)
                if cache is not None:
                    cache.clear()
                started = time.perf_counter()
                first = None
                total = 0
                async for pcm in tts.synthesize(text, settings.tts_voice_id, 1):
                    if first is None:
                        first = time.perf_counter() - started
                    total += len(pcm)
                elapsed = time.perf_counter() - started
                duration = total / 2 / 24_000
                firsts.append((first or 0) * 1000.0)
                totals.append(elapsed * 1000.0)
                durations.append(duration)
            duration = statistics.median(durations)
            row = {
                "text": text,
                "chars": len(text),
                "audio_s": round(duration, 2),
                "first_chunk_ms_p50": round(statistics.median(firsts), 1),
                "first_chunk_ms_max": round(max(firsts), 1),
                "total_ms_p50": round(statistics.median(totals), 1),
                "rtf_p50": round(statistics.median(totals) / max(1.0, duration * 1000), 3),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        health = tts.health() if hasattr(tts, "health") else {}
        summary = {
            "backend": args.backend,
            "warmup_s": round(warm_s, 2),
            "first_chunk_ms_p50": round(
                statistics.median([row["first_chunk_ms_p50"] for row in rows]), 1
            ),
            "first_chunk_ms_max": max(row["first_chunk_ms_max"] for row in rows),
            "rtf_p50": round(
                statistics.median([row["rtf_p50"] for row in rows]), 3
            ),
            "health": health,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.out:
            Path(args.out).write_text(
                json.dumps(
                    {"summary": summary, "rows": rows},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    finally:
        await tts.close()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="inprocess", choices=["inprocess", "vllm", "mock"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
