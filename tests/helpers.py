from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from echobridge.config import Settings
from echobridge.voices import VoiceRegistry, VoiceSpec


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
        # Never read the production voice registry from unit tests.
        tts_voice_dir="/nonexistent/echobridge-test-voices",
        tts_voice_registry="",
        tts_ref_audio="/nonexistent/echobridge-test-voices/reference.wav",
        tts_ref_text="/nonexistent/echobridge-test-voices/reference.txt",
    )
    base.update(overrides)
    return Settings(**base)


def write_wav(
    path: str | Path,
    *,
    sample_rate: int = 16_000,
    duration_ms: int = 20,
    frequency: float = 440.0,
    amplitude: float = 0.2,
) -> Path:
    """Write a tiny valid PCM16 mono WAV for registry/TTS fixtures."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(sample_rate * duration_ms / 1000))
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(frame_count)
    ]
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(struct.pack(f"<{frame_count}h", *samples))
    return target


def make_voice_registry(
    tmp_path: str | Path,
    *,
    ids: tuple[str, ...] = ("mingyue_v1", "mingyue_xixi_v1", "mingyue_nuonuo_v1"),
    default_id: str = "mingyue_v1",
) -> VoiceRegistry:
    """Create a registry whose reference files exist in ``tmp_path``."""
    base = Path(tmp_path)
    names = {
        "mingyue_v1": "月月",
        "mingyue_xixi_v1": "汐汐",
        "mingyue_nuonuo_v1": "糯糯",
    }
    specs = []
    for index, voice_id in enumerate(ids):
        audio = write_wav(
            base / f"{voice_id}.wav", frequency=220.0 + index * 110.0
        )
        text = base / f"{voice_id}.txt"
        text.write_text(f"参考文本 {voice_id}", encoding="utf-8")
        specs.append(
            VoiceSpec(
                id=voice_id,
                name=names.get(voice_id, voice_id),
                description=f"test voice {index}",
                ref_audio=audio,
                ref_text=text.read_text(encoding="utf-8"),
                default=voice_id == default_id,
            )
        )
    return VoiceRegistry(tuple(specs), default_id)


def loud_frame(amplitude: int = 9000) -> bytes:
    samples = 320
    values = []
    for i in range(samples):
        value = int(amplitude * math.sin(2 * math.pi * 220 * i / 16000))
        values.append(max(-32768, min(32767, value)))
    return struct.pack("<320h", *values)


def silence_frame() -> bytes:
    return b"\x00" * 640


def session_create(
    tools: list[dict] | None = None,
    *,
    voice: str | None = "mingyue_v1",
    include_voice: bool = True,
) -> dict:
    output = {
        "format": {"type": "pcm_s16le", "rate": 24000},
    }
    if include_voice:
        output["voice"] = voice if voice is not None else ""
    return {
        "type": "session.create",
        "event_id": "event_create",
        "session": {
            "model": "1.2.6.1",
            "instructions": "",
            "audio": {
                "input": {"format": {"type": "pcm", "rate": 16000}},
                "output": output,
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
