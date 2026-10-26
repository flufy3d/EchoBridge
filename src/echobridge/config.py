"""Runtime configuration for EchoBridge.

Phase 1 is intentionally boring: environment variables only.  Phase 6 can add a
TOML file without changing the rest of the gateway.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    bind: str = "127.0.0.1"
    port: int = 18787
    token: str = "dev-token"
    device_id: str = "x10a"

    # VAD / ASR.  "energy" and "mock" keep Phase 1 tests deterministic.
    vad_backend: str = "energy"
    vad_threshold: int = 700
    vad_start_frames: int = 2
    vad_endpoint_ms: int = 500
    vad_min_speech_ms: int = 200
    vad_max_speech_ms: int = 15_000
    vad_pre_roll_ms: int = 300
    vad_tail_pad_ms: int = 100
    vad_speculative_lead_ms: int = 150
    vad_start_ms: int = 64
    silero_threshold: float = 0.5
    silero_model: str | None = None

    asr_backend: str = "mock"
    asr_url: str = "http://127.0.0.1:18801"
    asr_timeout_s: float = 8.0
    mock_asr_text: str = "你好呀，溟月"

    tts_backend: str = "mock"
    tts_url: str = "http://127.0.0.1:18802"
    tts_timeout_s: float = 30.0
    tts_voice_id: str = "mingyue_v1"
    tts_ref_audio: str = "~/.cache/echobridge/voices/mingyue/reference.wav"
    tts_ref_text: str = "~/.cache/echobridge/voices/mingyue/reference.txt"
    tts_api_key: str = "EMPTY"

    # Mock LLM/TTS pacing.
    llm_char_delay_ms: int = 12
    tts_first_chunk_ms: int = 60
    tts_tone_char_ms: int = 55
    mock_tts_wav: str | None = None
    tool_timeout_s: float = 25.0

    # Dev-only probe page (static HTML; no secrets).
    enable_probe: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        dev = _env_bool("ECHOBRIDGE_DEV", True)
        token = os.environ.get("ECHOBRIDGE_TOKEN", "").strip()
        if not token:
            if not dev:
                raise RuntimeError(
                    "ECHOBRIDGE_TOKEN is required when ECHOBRIDGE_DEV is not enabled"
                )
            token = "dev-token"
        return cls(
            bind=os.environ.get("ECHOBRIDGE_BIND", "127.0.0.1").strip() or "127.0.0.1",
            port=_env_int("ECHOBRIDGE_PORT", 18787),
            token=token,
            device_id=os.environ.get("ECHOBRIDGE_DEVICE_ID", "x10a").strip() or "x10a",
            vad_backend=os.environ.get("ECHOBRIDGE_VAD_BACKEND", "energy").strip().lower(),
            vad_threshold=_env_int("ECHOBRIDGE_VAD_THRESHOLD", 700),
            vad_start_frames=_env_int("ECHOBRIDGE_VAD_START_FRAMES", 2),
            vad_endpoint_ms=_env_int("ECHOBRIDGE_VAD_ENDPOINT_MS", 500),
            vad_min_speech_ms=_env_int("ECHOBRIDGE_VAD_MIN_SPEECH_MS", 200),
            vad_max_speech_ms=_env_int("ECHOBRIDGE_VAD_MAX_SPEECH_MS", 15_000),
            vad_pre_roll_ms=_env_int("ECHOBRIDGE_VAD_PRE_ROLL_MS", 300),
            vad_tail_pad_ms=_env_int("ECHOBRIDGE_VAD_TAIL_PAD_MS", 100),
            vad_speculative_lead_ms=_env_int("ECHOBRIDGE_VAD_SPECULATIVE_LEAD_MS", 150),
            vad_start_ms=_env_int("ECHOBRIDGE_VAD_START_MS", 64),
            silero_threshold=float(os.environ.get("ECHOBRIDGE_SILERO_THRESHOLD", "0.5")),
            silero_model=os.environ.get("ECHOBRIDGE_SILERO_MODEL") or None,
            asr_backend=os.environ.get("ECHOBRIDGE_ASR_BACKEND", "mock").strip().lower(),
            asr_url=os.environ.get("ECHOBRIDGE_ASR_URL", "http://127.0.0.1:18801").rstrip("/"),
            asr_timeout_s=float(os.environ.get("ECHOBRIDGE_ASR_TIMEOUT_S", "8")),
            mock_asr_text=os.environ.get("ECHOBRIDGE_MOCK_ASR_TEXT", "你好呀，溟月"),
            tts_backend=os.environ.get("ECHOBRIDGE_TTS_BACKEND", "mock").strip().lower(),
            tts_url=os.environ.get("ECHOBRIDGE_TTS_URL", "http://127.0.0.1:18802").rstrip("/"),
            tts_timeout_s=float(os.environ.get("ECHOBRIDGE_TTS_TIMEOUT_S", "30")),
            tts_voice_id=os.environ.get("ECHOBRIDGE_TTS_VOICE_ID", "mingyue_v1").strip()
            or "mingyue_v1",
            tts_ref_audio=os.environ.get(
                "ECHOBRIDGE_TTS_REF_AUDIO",
                "~/.cache/echobridge/voices/mingyue/reference.wav",
            ),
            tts_ref_text=os.environ.get(
                "ECHOBRIDGE_TTS_REF_TEXT",
                "~/.cache/echobridge/voices/mingyue/reference.txt",
            ),
            tts_api_key=os.environ.get("ECHOBRIDGE_TTS_API_KEY", "EMPTY"),
            llm_char_delay_ms=_env_int("ECHOBRIDGE_LLM_CHAR_DELAY_MS", 12),
            tts_first_chunk_ms=_env_int("ECHOBRIDGE_TTS_FIRST_CHUNK_MS", 60),
            tts_tone_char_ms=_env_int("ECHOBRIDGE_TTS_TONE_CHAR_MS", 55),
            mock_tts_wav=os.environ.get("ECHOBRIDGE_MOCK_TTS_WAV") or None,
            tool_timeout_s=float(os.environ.get("ECHOBRIDGE_TOOL_TIMEOUT_S", "25")),
            enable_probe=_env_bool("ECHOBRIDGE_ENABLE_PROBE", True),
            log_level=os.environ.get("ECHOBRIDGE_LOG_LEVEL", "INFO").upper(),
        )
