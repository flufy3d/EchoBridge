#!/usr/bin/env python3
"""Qwen3-TTS worker benchmark: latency, audio duration, RTF. No playback."""

from __future__ import annotations

import argparse
import json
import statistics
import time

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
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rows = []
    with httpx.Client(timeout=120) as client:
        print(json.dumps(client.get(args.url + "/healthz").json(), ensure_ascii=False))
        for text in TEXTS:
            times = []
            last_headers = {}
            audio_bytes = 0
            for _ in range(args.repeat):
                started = time.perf_counter()
                response = client.post(
                    args.url + "/v1/synthesize",
                    json={"text": text, "voice_id": "mingyue_v1", "generation": 1},
                )
                elapsed = (time.perf_counter() - started) * 1000.0
                response.raise_for_status()
                times.append(elapsed)
                audio_bytes = len(response.content)
                last_headers = dict(response.headers)
            duration = audio_bytes / 2 / 24000
            rows.append(
                {
                    "text": text,
                    "chars": len(text),
                    "audio_s": round(duration, 2),
                    "first_packet_ms_p50": round(statistics.median(times), 1),
                    "first_packet_ms_max": round(max(times), 1),
                    "rtf_p50": round(statistics.median(times) / max(1.0, duration * 1000), 3),
                    "worker_ms": last_headers.get("x-latency-ms"),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
