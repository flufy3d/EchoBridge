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
            audio_bytes = 0
            for _ in range(args.repeat):
                started = time.perf_counter()
                first = None
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
                        if first is None:
                            first = time.perf_counter() - started
                        total += len(chunk)
                firsts.append((first or 0) * 1000.0)
                totals.append((time.perf_counter() - started) * 1000.0)
                audio_bytes = total
            duration = audio_bytes / 2 / 24000
            rows.append(
                {
                    "text": text,
                    "chars": len(text),
                    "audio_s": round(duration, 2),
                    "first_chunk_ms_p50": round(statistics.median(firsts), 1),
                    "first_chunk_ms_max": round(max(firsts), 1),
                    "total_ms_p50": round(statistics.median(totals), 1),
                    "rtf_p50": round(statistics.median(totals) / max(1.0, duration * 1000), 3),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
