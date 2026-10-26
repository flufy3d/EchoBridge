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

    # Mock ASR/VAD.  Phase 2 replaces the VAD and ASR adapter.
    vad_threshold: int = 700
    vad_start_frames: int = 2
    vad_endpoint_ms: int = 600
    vad_min_speech_ms: int = 200
    vad_max_speech_ms: int = 15_000
    mock_asr_text: str = "你好呀，溟月"

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
            vad_threshold=_env_int("ECHOBRIDGE_VAD_THRESHOLD", 700),
            vad_start_frames=_env_int("ECHOBRIDGE_VAD_START_FRAMES", 2),
            vad_endpoint_ms=_env_int("ECHOBRIDGE_VAD_ENDPOINT_MS", 600),
            vad_min_speech_ms=_env_int("ECHOBRIDGE_VAD_MIN_SPEECH_MS", 200),
            vad_max_speech_ms=_env_int("ECHOBRIDGE_VAD_MAX_SPEECH_MS", 15_000),
            mock_asr_text=os.environ.get("ECHOBRIDGE_MOCK_ASR_TEXT", "你好呀，溟月"),
            llm_char_delay_ms=_env_int("ECHOBRIDGE_LLM_CHAR_DELAY_MS", 12),
            tts_first_chunk_ms=_env_int("ECHOBRIDGE_TTS_FIRST_CHUNK_MS", 60),
            tts_tone_char_ms=_env_int("ECHOBRIDGE_TTS_TONE_CHAR_MS", 55),
            mock_tts_wav=os.environ.get("ECHOBRIDGE_MOCK_TTS_WAV") or None,
            tool_timeout_s=float(os.environ.get("ECHOBRIDGE_TOOL_TIMEOUT_S", "25")),
            enable_probe=_env_bool("ECHOBRIDGE_ENABLE_PROBE", True),
            log_level=os.environ.get("ECHOBRIDGE_LOG_LEVEL", "INFO").upper(),
        )
