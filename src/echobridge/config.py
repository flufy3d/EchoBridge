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


def _env_ints(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Read a comma-separated integer list (``1,2,4,8``)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated integer list") from exc
    return tuple(values) or default


def _env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return value


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
    # Long utterances get a shorter pre-roll in the ASR request: their leading
    # clause is less likely to be a soft onset, and 500 ms less audio saves
    # roughly 100 ms of encoder/generation time.  Short turns keep the full
    # pre-roll that protects soft leading words.
    vad_long_pre_roll_ms: int = 300
    vad_long_pre_roll_after_ms: int = 2500
    vad_tail_pad_ms: int = 100
    vad_speculative_lead_ms: int = 150
    # Long utterances spend proportionally longer in ASR.  Start their
    # speculative pass earlier; short turns keep the safe 150 ms lead so a
    # natural pause does not launch (and then discard) an expensive ASR run.
    vad_speculative_long_lead_ms: int = 350
    vad_speculative_long_after_ms: int = 2500
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
    # Debug: log mic level / max Silero probability once per second whenever
    # either is above the noise floor.
    vad_trace: bool = False
    silero_model: str | None = None
    # Force full-duplex input handling even when session.create does not
    # advertise extension.audio.input.full_duplex.  Used by tests and for
    # devices that have not shipped the extension yet.
    full_duplex: bool = False

    asr_backend: str = "mock"
    asr_url: str = "http://127.0.0.1:18801"
    asr_timeout_s: float = 8.0
    # In-process Qwen3-ASR (``ECHOBRIDGE_ASR_BACKEND=inprocess``).  The model
    # lives in the gateway process and shares its CUDA context with an
    # in-process TTS backend, if one is enabled.
    asr_model: str = "~/.cache/echobridge/models/Qwen3-ASR-0.6B"
    # Served model name when ``asr_backend=vllm`` (vLLM -m / --served-model-name).
    asr_vllm_model: str = "qwen3-asr"
    asr_language: str = "Chinese"
    asr_max_new_tokens: int = 192
    asr_device: str = "cuda:0"
    asr_attn_implementation: str = "sdpa"
    # Optional torch.compile for the ASR decoder.  Off by default: the first
    # call pays a long compilation cost and the gain on a 3070 is modest.
    asr_compile: bool = False
    mock_asr_text: str = "你好呀，溟月"
    # Qwen3-ASR context biasing.  This is *not* a weighted hotword table: the
    # terms are injected as the request context/system prompt, so keep the list
    # short and verify with real recordings.
    asr_hotwords: tuple[str, ...] = ()

    tts_backend: str = "mock"
    tts_url: str = "http://127.0.0.1:18802"
    tts_timeout_s: float = 30.0
    # After a TTS worker restart the gateway cache still holds the greeting,
    # so ensure_warm() would be a no-op.  Probe the worker and force one real
    # synthesis request on a down -> up edge.  0 disables the watcher.
    tts_recovery_probe_s: float = 10.0
    # In-process Qwen3-TTS (``ECHOBRIDGE_TTS_BACKEND=inprocess``).  The HF
    # implementation is experimental; vLLM-Omni remains the default when it is
    # available.  See docs/PHASE7.md for the measured comparison.
    tts_model: str = "~/.cache/echobridge/models/Qwen3-TTS-12Hz-0.6B-Base"
    tts_device: str = "cuda:0"
    tts_language: str = "Chinese"
    tts_max_new_tokens: int = 512
    tts_attn_implementation: str = "sdpa"
    tts_compile: bool = False
    # Streaming codec chunking, mirroring deploy/qwen3_tts_echobridge.yaml.
    # A flat 1 -> 25 frame step leaves 0.4-0.7 s of silence after the first
    # 80 ms packet, because the next packet waits for two seconds of audio.
    # Ramp the packet duration instead: 160, 320, 640, 960, 2000 ms.  The
    # first packet is 160 ms rather than 80 ms so playback does not run dry
    # before the second packet arrives.
    tts_initial_codec_frames: int = 1
    tts_codec_chunk_frames: int = 25
    tts_codec_chunk_ramp: tuple[int, ...] = (2, 4, 8, 12, 25)
    tts_codec_left_context_frames: int = 72
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
    # DeepSeek thinking is on by default and costs 200-400 ms before the first
    # spoken character.  Real-time voice replies do not need it; set to
    # "enabled" only for debugging a reasoning-heavy answer.
    llm_thinking: str = "disabled"
    # httpx closes idle keep-alive connections after 5 s by default, so almost
    # every turn paid a fresh TLS handshake.  Keep the pool warm between turns.
    llm_keepalive_s: float = 300.0
    # Open the TLS/HTTP connection when VAD hears the user start talking.
    llm_prewarm: bool = True
    # Start the LLM request as soon as the speculative ASR result is ready,
    # before endpoint confirmation.  Playback still waits for the endpoint.
    llm_speculative: bool = True

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
            vad_long_pre_roll_ms=_env_int("ECHOBRIDGE_VAD_LONG_PRE_ROLL_MS", 300),
            vad_long_pre_roll_after_ms=_env_int(
                "ECHOBRIDGE_VAD_LONG_PRE_ROLL_AFTER_MS", 2500
            ),
            vad_tail_pad_ms=_env_int("ECHOBRIDGE_VAD_TAIL_PAD_MS", 100),
            vad_speculative_lead_ms=_env_int("ECHOBRIDGE_VAD_SPECULATIVE_LEAD_MS", 150),
            vad_speculative_long_lead_ms=_env_int(
                "ECHOBRIDGE_VAD_SPECULATIVE_LONG_LEAD_MS", 350
            ),
            vad_speculative_long_after_ms=_env_int(
                "ECHOBRIDGE_VAD_SPECULATIVE_LONG_AFTER_MS", 2500
            ),
            vad_start_ms=_env_int("ECHOBRIDGE_VAD_START_MS", 64),
            silero_threshold=float(os.environ.get("ECHOBRIDGE_SILERO_THRESHOLD", "0.5")),
            silero_start_threshold=float(
                os.environ.get("ECHOBRIDGE_SILERO_START_THRESHOLD", "0.35")
            ),
            vad_start_gap_windows=_env_int("ECHOBRIDGE_VAD_START_GAP_WINDOWS", 1),
            vad_short_carry_ms=_env_int("ECHOBRIDGE_VAD_SHORT_CARRY_MS", 1000),
            vad_dump_dir=os.environ.get("ECHOBRIDGE_VAD_DUMP_DIR") or None,
            vad_trace=_env_bool("ECHOBRIDGE_VAD_TRACE", False),
            silero_model=os.environ.get("ECHOBRIDGE_SILERO_MODEL") or None,
            full_duplex=_env_bool("ECHOBRIDGE_FULL_DUPLEX", False),
            asr_backend=os.environ.get("ECHOBRIDGE_ASR_BACKEND", "mock").strip().lower(),
            asr_url=os.environ.get("ECHOBRIDGE_ASR_URL", "http://127.0.0.1:18801").rstrip("/"),
            asr_timeout_s=float(os.environ.get("ECHOBRIDGE_ASR_TIMEOUT_S", "8")),
            asr_model=os.environ.get(
                "ECHOBRIDGE_ASR_MODEL",
                "~/.cache/echobridge/models/Qwen3-ASR-0.6B",
            ),
            asr_vllm_model=os.environ.get(
                "ECHOBRIDGE_ASR_VLLM_MODEL", "qwen3-asr"
            ).strip()
            or "qwen3-asr",
            asr_language=os.environ.get("ECHOBRIDGE_ASR_LANGUAGE", "Chinese").strip(),
            asr_max_new_tokens=_env_int("ECHOBRIDGE_ASR_MAX_NEW_TOKENS", 192),
            asr_device=os.environ.get("ECHOBRIDGE_ASR_DEVICE", "cuda:0").strip()
            or "cuda:0",
            asr_attn_implementation=os.environ.get(
                "ECHOBRIDGE_ASR_ATTN_IMPLEMENTATION", "sdpa"
            ).strip()
            or "sdpa",
            asr_compile=_env_bool("ECHOBRIDGE_ASR_COMPILE", False),
            mock_asr_text=os.environ.get("ECHOBRIDGE_MOCK_ASR_TEXT", "你好呀，溟月"),
            asr_hotwords=_env_phrases("ECHOBRIDGE_ASR_HOTWORDS", ()),
            tts_backend=os.environ.get("ECHOBRIDGE_TTS_BACKEND", "mock").strip().lower(),
            tts_url=os.environ.get("ECHOBRIDGE_TTS_URL", "http://127.0.0.1:18802").rstrip("/"),
            tts_timeout_s=float(os.environ.get("ECHOBRIDGE_TTS_TIMEOUT_S", "30")),
            tts_recovery_probe_s=float(
                os.environ.get("ECHOBRIDGE_TTS_RECOVERY_PROBE_S", "10")
            ),
            tts_model=os.environ.get(
                "ECHOBRIDGE_TTS_MODEL",
                "~/.cache/echobridge/models/Qwen3-TTS-12Hz-0.6B-Base",
            ),
            tts_device=os.environ.get("ECHOBRIDGE_TTS_DEVICE", "cuda:0").strip()
            or "cuda:0",
            tts_language=os.environ.get("ECHOBRIDGE_TTS_LANGUAGE", "Chinese").strip()
            or "Chinese",
            tts_max_new_tokens=_env_int("ECHOBRIDGE_TTS_MAX_NEW_TOKENS", 512),
            tts_attn_implementation=os.environ.get(
                "ECHOBRIDGE_TTS_ATTN_IMPLEMENTATION", "sdpa"
            ).strip()
            or "sdpa",
            tts_compile=_env_bool("ECHOBRIDGE_TTS_COMPILE", False),
            tts_initial_codec_frames=_env_int("ECHOBRIDGE_TTS_INITIAL_CODEC_FRAMES", 1),
            tts_codec_chunk_frames=_env_int("ECHOBRIDGE_TTS_CODEC_CHUNK_FRAMES", 25),
            tts_codec_chunk_ramp=_env_ints(
                "ECHOBRIDGE_TTS_CODEC_CHUNK_RAMP", (2, 4, 8, 12, 25)
            ),
            tts_codec_left_context_frames=_env_int(
                "ECHOBRIDGE_TTS_CODEC_LEFT_CONTEXT_FRAMES", 72
            ),
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
            llm_thinking=_env_choice(
                "ECHOBRIDGE_LLM_THINKING", "disabled", {"disabled", "enabled"}
            ),
            llm_keepalive_s=float(os.environ.get("ECHOBRIDGE_LLM_KEEPALIVE_S", "300")),
            llm_prewarm=_env_bool("ECHOBRIDGE_LLM_PREWARM", True),
            llm_speculative=_env_bool("ECHOBRIDGE_LLM_SPECULATIVE", True),
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
