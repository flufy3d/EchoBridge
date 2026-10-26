#!/usr/bin/env python3
"""Silero VAD + Qwen3-ASR end-to-end latency benchmark (host only).

Feeds a 16 kHz WAV as 20 ms frames, measures VAD endpoint -> ASR final.  It does
not play audio on any device.

Run with the gateway venv:
  PYTHONPATH=src python scripts/bench_vad_asr.py \
      --wav ~/.cache/echobridge/bench/asr_zh.wav
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import wave

import httpx

from echobridge.config import Settings
from echobridge.vad import SileroVAD


def load_wav(path: str) -> bytes:
    with wave.open(path, "rb") as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise SystemExit("WAV must be mono PCM16")
        if handle.getframerate() != 16000:
            raise SystemExit("WAV must be 16 kHz")
        return handle.readframes(handle.getnframes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18801")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    pcm = load_wav(args.wav)
    settings = Settings(
        vad_backend="silero",
        vad_endpoint_ms=500,
        vad_min_speech_ms=200,
        vad_max_speech_ms=15000,
        vad_pre_roll_ms=300,
        vad_tail_pad_ms=100,
        vad_speculative_lead_ms=150,
        asr_backend="qwen3",
        asr_url=args.url,
    )
    vad = SileroVAD(settings)
    rows = []
    with httpx.Client(timeout=120) as client:
        for run in range(args.repeat):
            vad.reset()
            frames = [b"\x00\x00" * 320] * 15
            frames += [pcm[i : i + 640] for i in range(0, len(pcm), 640)]
            frames += [b"\x00\x00" * 320] * 60
            final = None
            endpoint_at = 0.0
            for index, frame in enumerate(frames):
                for event in vad.push(frame):
                    if event.kind == "final":
                        final = event
                        endpoint_at = time.perf_counter()
                        break
                if final is not None:
                    break
            if final is None:
                rows.append({"run": run + 1, "error": "no VAD endpoint"})
                continue
            started = time.perf_counter()
            response = client.post(
                args.url + "/v1/transcribe",
                content=final.pcm,
                headers={"X-Sample-Rate": "16000", "Content-Type": "application/octet-stream"},
            )
            response.raise_for_status()
            result = response.json()
            asr_ms = (time.perf_counter() - started) * 1000.0
            rows.append(
                {
                    "run": run + 1,
                    "audio_s": round(len(pcm) / 2 / 16000, 2),
                    "vad_speech_ms": final.speech_ms,
                    "vad_pcm_s": round(len(final.pcm) / 2 / 16000, 2),
                    "asr_wall_ms": round(asr_ms, 1),
                    "worker_ms": round(float(result.get("latency_ms", 0)), 1),
                    "text": result.get("text", ""),
                    "language": result.get("language", ""),
                }
            )
            print(json.dumps(rows[-1], ensure_ascii=False))

    if rows:
        values = [row["asr_wall_ms"] for row in rows if "asr_wall_ms" in row]
        if values:
            print(
                json.dumps(
                    {
                        "runs": len(values),
                        "asr_wall_ms_p50": round(statistics.median(values), 1),
                        "asr_wall_ms_max": round(max(values), 1),
                    },
                    ensure_ascii=False,
                )
            )
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
