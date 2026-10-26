#!/usr/bin/env python3
"""Protocol test client that mirrors the Android app's actual message shapes.

Examples:
  python scripts/ws_client.py --scenario all
  python scripts/ws_client.py --scenario turn --asr-text "溟月，笑一个"
  python scripts/ws_client.py --scenario cancel --url wss://host:8443/echobridge/ws
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import os
import struct
import sys
import urllib.parse
import wave
from pathlib import Path
from typing import Any

import websockets

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKEN_FILE = ROOT / "var" / "dev-token"


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
            "name": "play_animation",
            "description": "播放动作",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "enum": ["spin"]}},
                "required": ["name"],
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


def loud_frame() -> bytes:
    samples = 320
    values = [
        int(9000 * math.sin(2 * math.pi * 220 * i / 16000)) for i in range(samples)
    ]
    return struct.pack("<320h", *values)


def silence_frame() -> bytes:
    return b"\x00" * 640


def summarize(event: dict[str, Any]) -> str:
    kind = event.get("type", "?")
    if kind == "response.output_audio.delta":
        return f"{kind} response={event.get('response_id')} b64={len(event.get('delta', ''))}"
    if kind in ("response.output_text.delta",):
        return f"{kind} {event.get('delta', '')[:40]!r}"
    if kind in ("response.output_text.done",):
        return f"{kind} {event.get('text', '')[:60]!r}"
    if kind in (
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.delta",
    ):
        return f"{kind} {event.get('transcript', event.get('delta', ''))[:60]!r}"
    if kind == "response.function_call_arguments.done":
        return f"{kind} " + ",".join(
            item.get("name", "?") for item in event.get("items", [])
        )
    if kind == "response.done":
        return f"{kind} status={event.get('status')} response={event.get('response_id')}"
    if kind == "error":
        return f"{kind} {event.get('error')}"
    return kind


class Client:
    def __init__(self, ws, audio: bytearray) -> None:
        self.ws = ws
        self.events: list[dict[str, Any]] = []
        self.audio = audio
        self._reader = asyncio.create_task(self._read())

    async def _read(self) -> None:
        async for raw in self.ws:
            event = json.loads(raw)
            self.events.append(event)
            if event.get("type") == "response.output_audio.delta":
                self.audio.extend(base64.b64decode(event.get("delta", "")))
            print("  <-", summarize(event))

    async def send(self, event: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(event, ensure_ascii=False))

    async def wait_for(
        self, event_type: str, timeout: float = 10.0, after: int = 0
    ) -> dict[str, Any]:
        async def wait() -> dict[str, Any]:
            while True:
                for event in self.events[after:]:
                    if event.get("type") == event_type:
                        return event
                await asyncio.sleep(0.01)

        return await asyncio.wait_for(wait(), timeout=timeout)

    def index_of(self, event_type: str, response_id: str | None = None) -> int:
        for index, event in enumerate(self.events):
            if event.get("type") != event_type:
                continue
            if response_id is None or event.get("response_id") == response_id:
                return index
        raise AssertionError(f"{event_type} not found")

    async def greeting(self, text: str = "你来啦，我在呢。") -> dict[str, Any]:
        start = len(self.events)
        speech_id = "greet_client"
        split = len(text) // 2 or len(text)
        await self.send(
            {
                "type": "speech_text_buffer.append",
                "event_id": "event_greet_append",
                "speech_id": speech_id,
                "text": text[:split],
            }
        )
        await self.send(
            {
                "type": "speech_text_buffer.commit",
                "event_id": "event_greet_commit",
                "speech_id": speech_id,
                "text": text[split:],
            }
        )
        done = await self.wait_for("response.done", after=start)
        assert not any(
            event.get("type") == "response.output_text.delta"
            for event in self.events[start:]
        ), "greeting must not emit text deltas"
        return done

    async def turn(self, wav_pcm: bytes | None = None) -> dict[str, Any]:
        start = len(self.events)
        if wav_pcm:
            for offset in range(0, len(wav_pcm) - 639, 640):
                frame = wav_pcm[offset : offset + 640]
                await self.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame).decode("ascii"),
                    }
                )
                await asyncio.sleep(0.02)
        else:
            for _ in range(30):
                await self.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(loud_frame()).decode("ascii"),
                    }
                )
        for _ in range(40):
            await self.send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(silence_frame()).decode("ascii"),
                }
            )
            await asyncio.sleep(0.02)
        completed = await self.wait_for(
            "conversation.item.input_audio_transcription.completed", after=start
        )
        assert completed.get("transcript")
        handled: set[str] = set()
        while True:
            for event in list(self.events[start:]):
                if event.get("type") == "response.function_call_arguments.done":
                    items = event.get("items", [])
                    fresh = [item for item in items if item.get("call_id") not in handled]
                    if fresh:
                        for item in fresh:
                            handled.add(item["call_id"])
                        await self.send(
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
                            }
                        )
                        print(f"  -> tool result x{len(fresh)}")
                if event.get("type") == "response.done":
                    return event
            await asyncio.sleep(0.01)

    async def cancel_during_audio(self) -> None:
        start = len(self.events)
        long_text = "你来啦，我在呢。今天也要开开心心的呀。"
        speech_id = "cancel_check"
        split = len(long_text) // 2
        await self.send(
            {
                "type": "speech_text_buffer.append",
                "speech_id": speech_id,
                "text": long_text[:split],
            }
        )
        await self.send(
            {
                "type": "speech_text_buffer.commit",
                "speech_id": speech_id,
                "text": long_text[split:],
            }
        )
        started = await self.wait_for("response.output_audio.started", after=start)
        old_id = started["response_id"]
        await self.send({"type": "response.cancel"})
        canceled = await self.wait_for("response.canceled", after=start)
        assert canceled["response_id"] == old_id
        index = self.index_of("response.canceled", old_id)
        await asyncio.sleep(0.15)
        bad = [
            event
            for event in self.events[index + 1 :]
            if event.get("response_id") == old_id
            and event.get("type") in ("response.output_audio.delta", "response.done")
        ]
        assert not bad, f"stale events after cancel: {bad}"
        print("  cancel verified: no stale audio/response.done after canceled")

    async def close(self) -> None:
        await self.send({"type": "session.close", "event_id": "event_close"})
        await self.wait_for("session.closed")
        await asyncio.sleep(0.05)
        self._reader.cancel()
        await asyncio.gather(self._reader, return_exceptions=True)


async def run(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("ECHOBRIDGE_TOKEN", "")
    if not token and DEFAULT_TOKEN_FILE.exists():
        token = DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        print("token missing: --token, ECHOBRIDGE_TOKEN, or var/dev-token", file=sys.stderr)
        return 2

    url = args.url
    separator = "&" if "?" in url else "?"
    if args.asr_text:
        url += separator + "asr_text=" + urllib.parse.quote(args.asr_text)
        separator = "&"
    if not args.no_device_id:
        url += separator + "device_id=pc-client"

    headers = {
        "Authorization": f"Bearer {token}",
        "X-Device-Id": "pc-client",
    }
    audio = bytearray()
    print(f"connecting {args.url}")
    async with websockets.connect(
        url,
        additional_headers=headers,
        subprotocols=["echobridge"],
        open_timeout=5,
        close_timeout=2,
    ) as ws:
        client = Client(ws, audio)
        await client.send(session_create())
        created = await client.wait_for("session.created")
        print(f"session created: {created.get('session', {}).get('id')}")

        if args.scenario in ("greeting", "all"):
            print("scenario: greeting")
            await client.greeting()
            print("  greeting verified")

        if args.scenario in ("turn", "all"):
            print("scenario: turn")
            await client.turn(read_wav_pcm16_16k(args.wav) if args.wav else None)
            assert any(
                event.get("type") == "response.output_text.done" for event in client.events
            )
            print("  turn verified")

        if args.scenario == "tool":
            print("scenario: tool")
            tool_start = len(client.events)
            await client.turn(read_wav_pcm16_16k(args.wav) if args.wav else None)
            assert any(
                event.get("type") == "response.function_call_arguments.done"
                for event in client.events[tool_start:]
            ), "expected a tool call; pass --asr-text '溟月，笑一个'"
            print("  tool verified")

        if args.scenario in ("cancel", "all"):
            print("scenario: cancel")
            await client.cancel_during_audio()
            print("  cancel verified")

        print("scenario: close")
        await client.close()
        print("  close verified")

    if audio:
        out = Path(args.save_wav)
        out.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(bytes(audio))
        print(f"audio saved to {out} ({len(audio)} bytes)")
    return 0


def read_wav_pcm16_16k(path: str) -> bytes:
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2 or channels != 1 or rate != 16000:
        raise SystemExit("WAV must be mono PCM16 16 kHz")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", default="ws://127.0.0.1:18787/echobridge/ws", help="WebSocket URL"
    )
    parser.add_argument("--token", default="", help="EchoBridge token")
    parser.add_argument("--asr-text", default="", help="mock ASR transcript")
    parser.add_argument(
        "--scenario",
        choices=["greeting", "turn", "tool", "cancel", "all"],
        default="all",
    )
    parser.add_argument("--no-device-id", action="store_true")
    parser.add_argument("--save-wav", default="var/last_response.wav")
    parser.add_argument("--wav", default="", help="16 kHz mono PCM16 speech for turn scenarios")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
