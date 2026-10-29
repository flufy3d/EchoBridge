#!/usr/bin/env python3
"""Benchmark the vLLM-Omni Qwen3-TTS Base streaming endpoint. No playback."""

from __future__ import annotations

import argparse
import base64
import json
import os
import statistics
import time
from pathlib import Path

import httpx

TEXTS = [
    "你好呀，我在呢。",
    "今天也要开开心心的呀。",
    "现在是晚上十一点，该准备休息啦。",
    "我看看哦，这个大概是一个蓝色的杯子。",
    "23 乘以 47 等于 1081。",
]

SAMPLE_RATE = 24_000


def playout_stats(arrivals: list[tuple[float, int]]) -> dict[str, float]:
    """Simulate a no-buffer player and report audible starvation.

    The gateway paces 20 ms frames, but the upstream packet sizes are what
    determine whether the player ever runs dry.  ``stall_ms`` is the total
    time the player would have nothing to play if it started at the first
    packet; ``gap_after_first_ms`` isolates the 1-frame -> 25-frame cliff.
    """
    if not arrivals:
        return {
            "first_chunk_audio_ms": 0.0,
            "gap_after_first_ms": 0.0,
            "stall_ms": 0.0,
            "playout_ms": 0.0,
        }
    start, first_bytes = arrivals[0]
    first_audio_s = first_bytes / 2 / SAMPLE_RATE
    play_until = start + first_audio_s
    first_gap: float | None = None
    stall_s = 0.0
    for when, size in arrivals[1:]:
        duration = size / 2 / SAMPLE_RATE
        if when > play_until:
            gap = when - play_until
            stall_s += gap
            if first_gap is None:
                first_gap = gap
            play_until = when + duration
        else:
            play_until += duration
    return {
        "first_chunk_audio_ms": round(first_audio_s * 1000.0, 1),
        "gap_after_first_ms": round((first_gap or 0.0) * 1000.0, 1),
        "stall_ms": round(stall_s * 1000.0, 1),
        "playout_ms": round((play_until - start) * 1000.0, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18802")
    parser.add_argument("--ref-audio", default=str(Path.home() / ".cache/echobridge/voices/mingyue/reference.wav"))
    parser.add_argument("--ref-text", default=str(Path.home() / ".cache/echobridge/voices/mingyue/reference.txt"))
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    ref_audio = Path(os.path.expanduser(args.ref_audio))
    ref_text = Path(os.path.expanduser(args.ref_text)).read_text(encoding="utf-8").strip()
    ref_b64 = "data:audio/wav;base64," + base64.b64encode(ref_audio.read_bytes()).decode("ascii")
    rows = []
    with httpx.Client(timeout=120) as client:
        for text in TEXTS:
            payload = {
                "input": text,
                "task_type": "Base",
                "language": "Chinese",
                "response_format": "pcm",
                "stream": True,
                "stream_format": "audio",
                "ref_audio": ref_b64,
                "ref_text": ref_text,
            }
            firsts: list[float] = []
            totals: list[float] = []
            gaps: list[float] = []
            stalls: list[float] = []
            audio_bytes = 0
            playout = playout_stats([])
            for _ in range(args.repeat):
                started = time.perf_counter()
                arrivals: list[tuple[float, int]] = []
                total = 0
                with client.stream(
                    "POST",
                    args.url + "/v1/audio/speech",
                    json=payload,
                    headers={"Authorization": "Bearer EMPTY"},
                ) as response:
                    response.raise_for_status()
                    for chunk in response.iter_bytes():
                        if not chunk:
                            continue
                        arrivals.append((time.perf_counter() - started, len(chunk)))
                        total += len(chunk)
                firsts.append((arrivals[0][0] if arrivals else 0.0) * 1000.0)
                totals.append((time.perf_counter() - started) * 1000.0)
                playout = playout_stats(arrivals)
                gaps.append(playout["gap_after_first_ms"])
                stalls.append(playout["stall_ms"])
                audio_bytes = total
            duration = audio_bytes / 2 / 24000
            rows.append(
                {
                    "text": text,
                    "chars": len(text),
                    "audio_s": round(duration, 2),
                    "first_chunk_ms_p50": round(statistics.median(firsts), 1),
                    "first_chunk_ms_max": round(max(firsts), 1),
                    "first_chunk_audio_ms": playout["first_chunk_audio_ms"],
                    "gap_after_first_ms_p50": round(statistics.median(gaps), 1),
                    "gap_after_first_ms_max": round(max(gaps), 1),
                    "stall_ms_p50": round(statistics.median(stalls), 1),
                    "stall_ms_max": round(max(stalls), 1),
                    "total_ms_p50": round(statistics.median(totals), 1),
                    "rtf_p50": round(statistics.median(totals) / max(1.0, duration * 1000), 3),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
