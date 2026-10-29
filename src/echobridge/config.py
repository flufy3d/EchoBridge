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


def _env_phrases(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    phrases = tuple(part.strip() for part in raw.split("|") if part.strip())
    return phrases or default


def build_asr_context(hotwords: tuple[str, ...] | list[str]) -> str:
    """Build the free-form Qwen3-ASR context string from hotword terms."""
    terms = tuple(str(term).strip() for term in hotwords if str(term).strip())
    if not terms:
        return ""
    # A Chinese prefix and separators bias Chinese term spellings far better
    # than "Technical terms: ..." (verified with 溟月 at 8 dB SNR).
    return "专有名词：" + "、".join(terms)


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
    # Idle pre-roll must cover a short leading clause ("没什么，") plus the
    # comma pause that Silero misses on soft VOICE_COMMUNICATION audio.
    vad_pre_roll_ms: int = 800
    # While a response plays (full duplex) the pre-roll is AEC residual echo,
    # so keep it short.
    vad_playback_pre_roll_ms: int = 300
    vad_tail_pad_ms: int = 100
    vad_speculative_lead_ms: int = 150
    vad_start_ms: int = 64
    silero_threshold: float = 0.5
    # Lower threshold used only to *open* a segment while no response is
    # playing; continuation/endpointing keeps silero_threshold.
    silero_start_threshold: float = 0.35
    # Sub-threshold windows tolerated inside the start run before it resets.
    vad_start_gap_windows: int = 1
    # A segment dropped as too short ("停", then a pause) is prepended to the
    # next segment if that one starts within this window.
    vad_short_carry_ms: int = 1000
    # Debug only: write every final VAD segment (wav + transcript) here.
    vad_dump_dir: str | None = None
    silero_model: str | None = None
    # Force full-duplex input handling even when session.create does not
    # advertise extension.audio.input.full_duplex.  Used by tests and for
    # devices that have not shipped the extension yet.
    full_duplex: bool = False

    asr_backend: str = "mock"
    asr_url: str = "http://127.0.0.1:18801"
    asr_timeout_s: float = 8.0
    mock_asr_text: str = "你好呀，溟月"
    # Qwen3-ASR context biasing.  This is *not* a weighted hotword table: the
    # terms are injected as the request context/system prompt, so keep the list
    # short and verify with real recordings.
    asr_hotwords: tuple[str, ...] = ()

    tts_backend: str = "mock"
    tts_url: str = "http://127.0.0.1:18802"
    tts_timeout_s: float = 30.0
    tts_voice_id: str = "mingyue_v1"
    # Voice registry.  ``tts_voice_registry`` is optional; when empty the
    # gateway uses ``<tts_voice_dir>/voices.json``.  The legacy ref_audio/
    # ref_text pair remains the fallback when the registry is missing or bad.
    tts_voice_dir: str = "~/.cache/echobridge/voices/mingyue"
    tts_voice_registry: str = ""
    tts_ref_audio: str = "~/.cache/echobridge/voices/mingyue/reference.wav"
    tts_ref_text: str = "~/.cache/echobridge/voices/mingyue/reference.txt"
    tts_api_key: str = "EMPTY"

    llm_backend: str = "mock"
    llm_url: str = "https://opencode.ai/inference/openai/v1/chat/completions"
    llm_model: str = "deepseek-v4.1-flash"
    llm_api_key: str = ""
    llm_max_tokens: int = 400
    llm_temperature: float = 0.6
    llm_first_token_timeout_s: float = 8.0
    llm_total_timeout_s: float = 30.0

    # Mock LLM/TTS pacing.
    llm_char_delay_ms: int = 12
    tts_first_chunk_ms: int = 60
    tts_tone_char_ms: int = 55
    mock_tts_wav: str | None = None
    tool_timeout_s: float = 25.0

    # Natural backchannel ("我想想～") played when the real answer is slow.  It
    # is cached at TTS warmup and sent as a separate response.backchannel event
    # so it never shares the formal response's pacer/queue.  Formal audio never
    # starts within 0.7 s and half the turns start within 1.5 s, so a shorter
    # delay makes the filler fire on almost every turn and delays fast answers.
    backchannel_enabled: bool = True
    backchannel_delay_ms: int = 1500
    backchannel_phrases: tuple[str, ...] = (
        "嗯…让我想想",
        "我想想～",
        "这个嘛～",
        "我看看哦～",
    )
    # Hand-picked recordings: ``<dir>/<voice_id>/clips.json`` maps phrase ->
    # WAV file.  A phrase with a clip skips TTS synthesis for that voice (one
    # synthesis per warmup varies a lot between takes).  Empty means
    # ``<tts_voice_dir>/backchannel``.
    backchannel_clip_dir: str = ""

    # Dev-only probe page (static HTML; no secrets).
    enable_probe: bool = True
    log_level: str = "INFO"

    @property
    def asr_context(self) -> str:
        """Context string sent to Qwen3-ASR for contextual biasing.

        The open-source Qwen3-ASR package has no weighted vocabulary table;
        ``context`` is a free-form prompt.  A short Chinese "专有名词：" prefix
        biases Chinese term spellings better than the old English prefix;
        keep the list short.
        """
        return build_asr_context(self.asr_hotwords)

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
            vad_pre_roll_ms=_env_int("ECHOBRIDGE_VAD_PRE_ROLL_MS", 800),
            vad_playback_pre_roll_ms=_env_int("ECHOBRIDGE_VAD_PLAYBACK_PRE_ROLL_MS", 300),
            vad_tail_pad_ms=_env_int("ECHOBRIDGE_VAD_TAIL_PAD_MS", 100),
            vad_speculative_lead_ms=_env_int("ECHOBRIDGE_VAD_SPECULATIVE_LEAD_MS", 150),
            vad_start_ms=_env_int("ECHOBRIDGE_VAD_START_MS", 64),
            silero_threshold=float(os.environ.get("ECHOBRIDGE_SILERO_THRESHOLD", "0.5")),
            silero_start_threshold=float(
                os.environ.get("ECHOBRIDGE_SILERO_START_THRESHOLD", "0.35")
            ),
            vad_start_gap_windows=_env_int("ECHOBRIDGE_VAD_START_GAP_WINDOWS", 1),
            vad_short_carry_ms=_env_int("ECHOBRIDGE_VAD_SHORT_CARRY_MS", 1000),
            vad_dump_dir=os.environ.get("ECHOBRIDGE_VAD_DUMP_DIR") or None,
            silero_model=os.environ.get("ECHOBRIDGE_SILERO_MODEL") or None,
            full_duplex=_env_bool("ECHOBRIDGE_FULL_DUPLEX", False),
            asr_backend=os.environ.get("ECHOBRIDGE_ASR_BACKEND", "mock").strip().lower(),
            asr_url=os.environ.get("ECHOBRIDGE_ASR_URL", "http://127.0.0.1:18801").rstrip("/"),
            asr_timeout_s=float(os.environ.get("ECHOBRIDGE_ASR_TIMEOUT_S", "8")),
            mock_asr_text=os.environ.get("ECHOBRIDGE_MOCK_ASR_TEXT", "你好呀，溟月"),
            asr_hotwords=_env_phrases("ECHOBRIDGE_ASR_HOTWORDS", ()),
            tts_backend=os.environ.get("ECHOBRIDGE_TTS_BACKEND", "mock").strip().lower(),
            tts_url=os.environ.get("ECHOBRIDGE_TTS_URL", "http://127.0.0.1:18802").rstrip("/"),
            tts_timeout_s=float(os.environ.get("ECHOBRIDGE_TTS_TIMEOUT_S", "30")),
            tts_voice_id=os.environ.get("ECHOBRIDGE_TTS_VOICE_ID", "mingyue_v1").strip()
            or "mingyue_v1",
            tts_voice_dir=os.environ.get(
                "ECHOBRIDGE_TTS_VOICE_DIR",
                "~/.cache/echobridge/voices/mingyue",
            ).strip()
            or "~/.cache/echobridge/voices/mingyue",
            tts_voice_registry=os.environ.get(
                "ECHOBRIDGE_TTS_VOICE_REGISTRY", ""
            ).strip(),
            tts_ref_audio=os.environ.get(
                "ECHOBRIDGE_TTS_REF_AUDIO",
                "~/.cache/echobridge/voices/mingyue/reference.wav",
            ),
            tts_ref_text=os.environ.get(
                "ECHOBRIDGE_TTS_REF_TEXT",
                "~/.cache/echobridge/voices/mingyue/reference.txt",
            ),
            tts_api_key=os.environ.get("ECHOBRIDGE_TTS_API_KEY", "EMPTY"),
            llm_backend=os.environ.get("ECHOBRIDGE_LLM_BACKEND", "mock").strip().lower(),
            llm_url=os.environ.get(
                "ECHOBRIDGE_LLM_URL",
                "https://opencode.ai/inference/openai/v1/chat/completions",
            ),
            llm_model=os.environ.get("ECHOBRIDGE_LLM_MODEL", "deepseek-v4.1-flash").strip(),
            llm_api_key=os.environ.get("ECHOBRIDGE_LLM_API_KEY", "").strip(),
            llm_max_tokens=_env_int("ECHOBRIDGE_LLM_MAX_TOKENS", 400),
            llm_temperature=float(os.environ.get("ECHOBRIDGE_LLM_TEMPERATURE", "0.6")),
            llm_first_token_timeout_s=float(
                os.environ.get("ECHOBRIDGE_LLM_FIRST_TOKEN_TIMEOUT_S", "8")
            ),
            llm_total_timeout_s=float(os.environ.get("ECHOBRIDGE_LLM_TOTAL_TIMEOUT_S", "30")),
            llm_char_delay_ms=_env_int("ECHOBRIDGE_LLM_CHAR_DELAY_MS", 12),
            tts_first_chunk_ms=_env_int("ECHOBRIDGE_TTS_FIRST_CHUNK_MS", 60),
            tts_tone_char_ms=_env_int("ECHOBRIDGE_TTS_TONE_CHAR_MS", 55),
            mock_tts_wav=os.environ.get("ECHOBRIDGE_MOCK_TTS_WAV") or None,
            tool_timeout_s=float(os.environ.get("ECHOBRIDGE_TOOL_TIMEOUT_S", "25")),
            backchannel_enabled=_env_bool("ECHOBRIDGE_BACKCHANNEL_ENABLED", True),
            backchannel_delay_ms=_env_int("ECHOBRIDGE_BACKCHANNEL_DELAY_MS", 1500),
            backchannel_phrases=_env_phrases(
                "ECHOBRIDGE_BACKCHANNEL_PHRASES",
                ("嗯…让我想想", "我想想～", "这个嘛～", "我看看哦～"),
            ),
            backchannel_clip_dir=os.environ.get(
                "ECHOBRIDGE_BACKCHANNEL_CLIP_DIR", ""
            ).strip(),
            enable_probe=_env_bool("ECHOBRIDGE_ENABLE_PROBE", True),
            log_level=os.environ.get("ECHOBRIDGE_LOG_LEVEL", "INFO").upper(),
        )
