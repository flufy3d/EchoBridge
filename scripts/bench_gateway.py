#!/usr/bin/env python3
"""Phase 6 gateway benchmark: 30 real ASR/TTS/LLM turns over one WSS session.

Measures the same client-side points as the Android app:
  * last speech frame -> transcription.completed (endpoint + ASR)
  * transcription.completed -> first formal audio delta
  * transcription.completed -> first perceived audio (backchannel or formal)
  * response.cancel -> response.canceled

This is not a 1 h/24 h soak.  It is the short acceptance run described in
docs/BENCHMARK.md; long-running behaviour is covered by the watchdog and by
normal daily use.
"""

from __future__ import annotations

import argparse
import array
import asyncio
import base64
import json
import math
import os
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import Any

import websockets

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_FILE = ROOT / "var" / "dev-token"
DEFAULT_WAVS = [
    "var/phase5_question_16k.wav",
    "var/phase5_turn.wav",
    "var/phase5_turn3.wav",
    "var/phase5_turn4.wav",
]


def make_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "set_emotion",
            "description": "改变表情",
            "parameters": {
                "type": "object",
                "properties": {"emotion": {"type": "string", "enum": ["happy", "sad"]}},
                "required": ["emotion"],
            },
        },
        {
            "type": "function",
            "name": "get_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def session_create() -> dict[str, Any]:
    return {
        "type": "session.create",
        "event_id": "event_create",
        "session": {
            "model": "1.2.6.1",
            "instructions": "",
            "audio": {
                "input": {"format": {"type": "pcm", "rate": 16000}},
                "output": {
                    "format": {"type": "pcm_s16le", "rate": 24000},
                    "voice": "mingyue_v1",
                },
            },
            "tools": make_tools(),
        },
        "extension": {
            "asr": {"extra": {}},
            "tts": {"extra": {}},
            "dialog": {"extra": {"enable_loudness_norm": True}},
        },
    }


def resample_pcm16(frames: bytes, source_rate: int) -> bytes:
    if source_rate == 16000:
        return frames
    try:
        import audioop

        converted, _ = audioop.ratecv(frames, 2, 1, source_rate, 16000, None)
        return converted
    except ImportError:
        pass
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - (len(frames) % 2)])
    source_count = len(samples)
    target_count = max(1, int(source_count * 16000 / source_rate))
    out = array.array("h")
    for index in range(target_count):
        position = index * source_rate / 16000
        left = int(position)
        right = min(left + 1, source_count - 1)
        fraction = position - left
        out.append(int(samples[left] * (1 - fraction) + samples[right] * fraction))
    return out.tobytes()


def read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels != 1:
        raise SystemExit(f"WAV must be mono PCM16: {path}")
    if len(frames) % 2:
        frames = frames[:-1]
    frames = resample_pcm16(frames, rate)
    return trim_silence(frames), 16000


def trim_silence(frames: bytes, threshold: int = 500) -> bytes:
    samples = array.array("h")
    samples.frombytes(frames)
    if not samples:
        return frames
    first = 0
    last = len(samples) - 1
    while first < len(samples) and abs(samples[first]) < threshold:
        first += 1
    while last > first and abs(samples[last]) < threshold:
        last -= 1
    pad = 1600  # 100 ms of context on each side
    first = max(0, first - pad)
    last = min(len(samples) - 1, last + pad)
    if last <= first:
        return frames
    return samples[first : last + 1].tobytes()


def limit_speech(frames: bytes, max_ms: int) -> bytes:
    if max_ms <= 0:
        return frames
    max_bytes = int(16000 * max_ms / 1000) * 2
    return frames[:max_bytes] if len(frames) > max_bytes else frames


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p / 100.0 * len(ordered)) - 1)
    return ordered[index]


def stats(values: list[float]) -> dict[str, float | None]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "n": len(present),
        "mean": round(statistics.mean(present), 1),
        "p50": round(percentile(present, 50) or 0.0, 1),
        "p90": round(percentile(present, 90) or 0.0, 1),
        "p95": round(percentile(present, 95) or 0.0, 1),
        "max": round(max(present), 1),
    }


class TurnRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.t_speech_end = 0.0
        self.t_completed = 0.0
        self.t_started = 0.0
        self.t_first_text = 0.0
        self.t_first_audio = 0.0
        self.t_backchannel = 0.0
        self.t_done = 0.0
        self.response_id = ""
        self.done = asyncio.Event()
        self.tool_hops = 0
        self.transcript_chars = 0
        self.transcription_failed = False
        self.error = ""
        self._handled_calls: set[str] = set()
        self._ws: Any = None

    async def read_loop(self) -> None:
        try:
            async for raw in self._ws:
                now = time.perf_counter()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self.events.append(event)
                kind = event.get("type", "")
                if kind == "conversation.item.input_audio_transcription.started":
                    self.t_started = self.t_started or now
                elif kind == "conversation.item.input_audio_transcription.completed":
                    self.t_completed = now
                    self.transcript_chars = len(event.get("transcript", ""))
                elif kind == "conversation.item.input_audio_transcription.failed":
                    self.transcription_failed = True
                    self.done.set()
                elif kind == "response.output_text.delta":
                    self.t_first_text = self.t_first_text or now
                elif kind == "response.backchannel":
                    self.t_backchannel = self.t_backchannel or now
                elif kind == "response.output_audio.started":
                    self.t_first_audio = self.t_first_audio or now
                elif kind == "response.output_audio.delta":
                    self.t_first_audio = self.t_first_audio or now
                elif kind == "response.function_call_arguments.done":
                    await self._send_tool_results(event)
                elif kind == "response.done":
                    if event.get("status") == "tool_calls":
                        self.tool_hops += 1
                        # The continuation response has new metrics.
                        self.t_first_text = 0.0
                        self.t_first_audio = 0.0
                        self.response_id = event.get("response_id", self.response_id)
                        continue
                    self.t_done = now
                    self.response_id = event.get("response_id", self.response_id)
                    self.done.set()
                elif kind == "error":
                    self.error = str(event.get("error") or event)
                    self.done.set()
        except Exception:
            self.done.set()

    async def _send_tool_results(self, event: dict[str, Any]) -> None:
        fresh = [
            item
            for item in event.get("items", [])
            if item.get("call_id") and item["call_id"] not in self._handled_calls
        ]
        if not fresh:
            return
        for item in fresh:
            self._handled_calls.add(item["call_id"])
        await self._ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "event_id": "event_tool_result",
                    "items": [
                        {
                            "type": "message",
                            "role": "tool",
                            "call_id": item["call_id"],
                            "content": [{"type": "input_text", "text": "ok"}],
                        }
                        for item in fresh
                    ],
                },
                ensure_ascii=False,
            )
        )

    def result(self) -> dict[str, Any]:
        def after_asr(value: float) -> float | None:
            if not value or not self.t_completed:
                return None
            return round((value - self.t_completed) * 1000.0, 1)

        formal = after_asr(self.t_first_audio)
        backchannel = after_asr(self.t_backchannel)
        perceived_candidates = [value for value in (backchannel, formal) if value is not None]
        perceived = min(perceived_candidates) if perceived_candidates else None
        return {
            "transcript_chars": self.transcript_chars,
            "tool_hops": self.tool_hops,
            "asr_final_from_speech_ms": round(
                (self.t_completed - self.t_speech_end) * 1000.0, 1
            )
            if self.t_completed and self.t_speech_end
            else None,
            "first_text_after_asr_ms": after_asr(self.t_first_text),
            "first_audio_after_asr_ms": formal,
            "backchannel_after_asr_ms": backchannel,
            "perceived_after_asr_ms": perceived,
            "total_ms": round((self.t_done - self.t_speech_end) * 1000.0, 1)
            if self.t_done and self.t_speech_end
            else None,
            "response_id": self.response_id,
        }


async def run_turn(ws: Any, pcm: bytes, timeout: float, endpoint_ms: int) -> dict[str, Any]:
    recorder = TurnRecorder()
    recorder._ws = ws
    reader = asyncio.create_task(recorder.read_loop())
    try:
        start = time.perf_counter()
        last_speech = start
        for offset in range(0, len(pcm) - 639, 640):
            frame = pcm[offset : offset + 640]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame).decode("ascii"),
                    }
                )
            )
            now = time.perf_counter()
            if max(abs(value) for value in array.array("h", frame)) > 500:
                last_speech = now
            await asyncio.sleep(0.02)
        recorder.t_speech_end = last_speech
        # 800 ms trailing silence: 500 ms endpoint plus margin.
        for _ in range(40):
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(b"\x00" * 640).decode("ascii"),
                    }
                )
            )
            await asyncio.sleep(0.02)
        await asyncio.wait_for(recorder.done.wait(), timeout=timeout)
        if recorder.transcription_failed:
            raise RuntimeError("transcription.failed (empty/short ASR)")
        if recorder.error:
            raise RuntimeError(f"gateway error: {recorder.error}")
        if not recorder.t_completed:
            raise RuntimeError("no transcription.completed")
        result = recorder.result()
        if result["asr_final_from_speech_ms"] is not None:
            result["asr_after_endpoint_est_ms"] = round(
                max(0.0, result["asr_final_from_speech_ms"] - endpoint_ms), 1
            )
        return result
    finally:
        reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)


async def run_cancel(ws: Any, index: int, timeout: float) -> dict[str, Any]:
    text = f"这是第{index}次打断测试，我故意说一句比较长的话，让正式音频有机会开始播放之后再取消。"
    speech_id = f"cancel_{index}"
    split = len(text) // 2
    start = time.perf_counter()
    await ws.send(
        json.dumps(
            {
                "type": "speech_text_buffer.append",
                "speech_id": speech_id,
                "text": text[:split],
            },
            ensure_ascii=False,
        )
    )
    await ws.send(
        json.dumps(
            {
                "type": "speech_text_buffer.commit",
                "speech_id": speech_id,
                "text": text[split:],
            },
            ensure_ascii=False,
        )
    )

    started_at = 0.0
    canceled_at = 0.0
    response_id = ""
    stale = 0
    events: list[dict[str, Any]] = []
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.perf_counter()))
        except asyncio.TimeoutError:
            break
        event = json.loads(raw)
        events.append(event)
        now = time.perf_counter()
        if event.get("type") == "response.output_audio.started" and not started_at:
            started_at = now
            response_id = event.get("response_id", "")
            await ws.send(json.dumps({"type": "response.cancel"}))
        elif event.get("type") == "response.canceled" and started_at:
            canceled_at = now
            break
    if not started_at or not canceled_at:
        return {"ok": False, "reason": "audio/canceled timeout"}
    index_canceled = next(
        (
            i
            for i, event in enumerate(events)
            if event.get("type") == "response.canceled"
            and event.get("response_id") == response_id
        ),
        -1,
    )
    if index_canceled >= 0:
        await asyncio.sleep(0.6)
        # Drain late frames so stale audio has a chance to appear.
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                event = json.loads(raw)
                events.append(event)
                if (
                    event.get("type") == "response.output_audio.delta"
                    and event.get("response_id") == response_id
                ):
                    stale += 1
        except asyncio.TimeoutError:
            pass
    return {
        "ok": stale == 0,
        "cancel_ms": round((canceled_at - started_at) * 1000.0, 1),
        "stale_audio_after_cancel": stale,
        "response_id": response_id,
    }


async def run(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("ECHOBRIDGE_TOKEN", "")
    if not token and DEFAULT_TOKEN_FILE.exists():
        token = DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        print("token missing: --token, ECHOBRIDGE_TOKEN, or var/dev-token", file=sys.stderr)
        return 2

    wav_args = args.wav or DEFAULT_WAVS
    wavs: list[tuple[str, bytes]] = []
    for item in wav_args:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            print(f"missing wav: {path}", file=sys.stderr)
            continue
        pcm, _ = read_wav(path)
        pcm = limit_speech(pcm, args.max_speech_ms)
        wavs.append((f"{path.name}@{len(pcm) // 32}ms", pcm))
    if not wavs:
        print("no input wavs", file=sys.stderr)
        return 2

    url = args.url
    url += ("&" if "?" in url else "?") + "device_id=bench-pc"
    headers = {"Authorization": f"Bearer {token}", "X-Device-Id": "bench-pc"}
    turns: list[dict[str, Any]] = []
    greeting: dict[str, Any] = {}
    cancellations: list[dict[str, Any]] = []

    print(f"connecting {args.url} · {len(wavs)} wavs · {args.turns} turns")
    async with websockets.connect(
        url,
        additional_headers=headers,
        subprotocols=["echobridge"],
        open_timeout=5,
        close_timeout=2,
    ) as ws:
        await ws.send(json.dumps(session_create(), ensure_ascii=False))
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            event = json.loads(raw)
            if event.get("type") == "session.created":
                print(f"session: {event.get('session', {}).get('id')}")
                break
            if event.get("type") == "error":
                raise RuntimeError(event)

        # Greeting: same split append/commit shape as the app.
        greet_start = time.perf_counter()
        text = "你来啦，我在呢。"
        split = len(text) // 2
        await ws.send(
            json.dumps(
                {
                    "type": "speech_text_buffer.append",
                    "speech_id": "bench_greet",
                    "text": text[:split],
                },
                ensure_ascii=False,
            )
        )
        commit_at = time.perf_counter()
        await ws.send(
            json.dumps(
                {
                    "type": "speech_text_buffer.commit",
                    "speech_id": "bench_greet",
                    "text": text[split:],
                },
                ensure_ascii=False,
            )
        )
        greet_audio = 0.0
        greet_done = 0.0
        deadline = time.perf_counter() + 15
        while time.perf_counter() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            event = json.loads(raw)
            now = time.perf_counter()
            if event.get("type") in (
                "response.output_audio.started",
                "response.output_audio.delta",
            ):
                greet_audio = greet_audio or now
            if event.get("type") == "response.done":
                greet_done = now
                break
        greeting = {
            "first_audio_after_commit_ms": round((greet_audio - commit_at) * 1000.0, 1)
            if greet_audio
            else None,
            "total_ms": round((greet_done - greet_start) * 1000.0, 1)
            if greet_done
            else None,
        }
        print(f"greeting: first_audio={greeting['first_audio_after_commit_ms']} ms")

        for index in range(args.turns):
            name, pcm = wavs[index % len(wavs)]
            try:
                result = await asyncio.wait_for(
                    run_turn(ws, pcm, args.timeout, args.endpoint_ms),
                    timeout=args.timeout + 15,
                )
                result.update({"turn": index + 1, "wav": name})
                turns.append(result)
                print(
                    f"{index + 1:02d}/{args.turns} {name} "
                    f"asr={result['asr_final_from_speech_ms']} ms "
                    f"formal={result['first_audio_after_asr_ms']} ms "
                    f"perceived={result['perceived_after_asr_ms']} ms "
                    f"total={result['total_ms']} ms"
                )
            except Exception as exc:
                turns.append(
                    {
                        "turn": index + 1,
                        "wav": name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"{index + 1:02d}/{args.turns} ERROR {type(exc).__name__}: {exc}")
            await asyncio.sleep(args.pause)

        for cancel_index in range(1, args.cancel_samples + 1):
            try:
                result = await run_cancel(ws, cancel_index, args.timeout)
            except Exception as exc:
                result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            cancellations.append(result)
            print(f"cancel {cancel_index}: {result}")
            await asyncio.sleep(0.3)

        await ws.send(json.dumps({"type": "session.close", "event_id": "event_close"}))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=2)
                if json.loads(raw).get("type") == "session.closed":
                    break
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            pass

    ok_turns = [turn for turn in turns if "error" not in turn]
    summary = {
        "turns_requested": args.turns,
        "turns_ok": len(ok_turns),
        "turns_failed": len(turns) - len(ok_turns),
        "asr_final_from_speech_ms": stats(
            [turn.get("asr_final_from_speech_ms") for turn in ok_turns]
        ),
        "asr_after_endpoint_est_ms": stats(
            [turn.get("asr_after_endpoint_est_ms") for turn in ok_turns]
        ),
        "first_text_after_asr_ms": stats(
            [turn.get("first_text_after_asr_ms") for turn in ok_turns]
        ),
        "first_audio_after_asr_ms": stats(
            [turn.get("first_audio_after_asr_ms") for turn in ok_turns]
        ),
        "backchannel_after_asr_ms": stats(
            [turn.get("backchannel_after_asr_ms") for turn in ok_turns]
        ),
        "perceived_after_asr_ms": stats(
            [turn.get("perceived_after_asr_ms") for turn in ok_turns]
        ),
        "total_ms": stats([turn.get("total_ms") for turn in ok_turns]),
        "endpoint_to_formal_audio_ms": stats(
            [
                turn["asr_after_endpoint_est_ms"] + turn["first_audio_after_asr_ms"]
                for turn in ok_turns
                if turn.get("asr_after_endpoint_est_ms") is not None
                and turn.get("first_audio_after_asr_ms") is not None
            ]
        ),
        "endpoint_to_perceived_audio_ms": stats(
            [
                turn["asr_after_endpoint_est_ms"] + turn["perceived_after_asr_ms"]
                for turn in ok_turns
                if turn.get("asr_after_endpoint_est_ms") is not None
                and turn.get("perceived_after_asr_ms") is not None
            ]
        ),
        "speech_end_to_formal_audio_ms": stats(
            [
                turn["asr_final_from_speech_ms"] + turn["first_audio_after_asr_ms"]
                for turn in ok_turns
                if turn.get("asr_final_from_speech_ms") is not None
                and turn.get("first_audio_after_asr_ms") is not None
            ]
        ),
        "greeting": greeting,
        "cancellations": cancellations,
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "url": args.url,
        "turns": turns,
        "summary": summary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["turns_failed"] == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:18787/echobridge/ws")
    parser.add_argument("--token", default="")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--wav", action="append", default=[])
    parser.add_argument("--out", default="var/bench/phase6.json")
    parser.add_argument("--timeout", type=float, default=35.0)
    parser.add_argument("--pause", type=float, default=0.3)
    parser.add_argument("--endpoint-ms", type=int, default=500)
    parser.add_argument("--max-speech-ms", type=int, default=4000)
    parser.add_argument("--cancel-samples", type=int, default=5)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
