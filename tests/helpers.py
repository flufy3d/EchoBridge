from __future__ import annotations

import math
import struct

from echobridge.config import Settings


def make_settings(**overrides) -> Settings:
    base = dict(
        token="test-token",
        vad_threshold=700,
        vad_start_frames=2,
        vad_endpoint_ms=600,
        vad_min_speech_ms=200,
        vad_max_speech_ms=15_000,
        llm_char_delay_ms=0,
        tts_first_chunk_ms=1,
        tts_tone_char_ms=8,
        tool_timeout_s=3.0,
        enable_probe=True,
    )
    base.update(overrides)
    return Settings(**base)


def loud_frame(amplitude: int = 9000) -> bytes:
    samples = 320
    values = []
    for i in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * 220 * i / 16000))
        values.append(max(-32768, min(32767, value)))
    return struct.pack("<320h", *values)


def silence_frame() -> bytes:
    return b"\x00" * 640


def session_create(tools: list[dict] | None = None) -> dict:
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
            "tools": tools or [],
        },
        "extension": {
            "asr": {"extra": {}},
            "tts": {"extra": {}},
            "dialog": {"extra": {"enable_loudness_norm": True}},
        },
    }


def set_emotion_tool() -> dict:
    return {
        "type": "function",
        "name": "set_emotion",
        "description": "change expression",
        "parameters": {
            "type": "object",
            "properties": {"emotion": {"type": "string", "enum": ["happy", "sad"]}},
            "required": ["emotion"],
        },
    }


def play_animation_tool() -> dict:
    return {
        "type": "function",
        "name": "play_animation",
        "description": "play animation",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string", "enum": ["spin"]}},
            "required": ["name"],
        },
    }


def get_time_tool() -> dict:
    return {
        "type": "function",
        "name": "get_time",
        "description": "current time",
        "parameters": {"type": "object", "properties": {}},
    }


def ask_expert_tool() -> dict:
    return {
        "type": "function",
        "name": "ask_expert",
        "description": "expert",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    }
