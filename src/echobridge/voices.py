"""Voice registry for the EchoBridge TTS backends.

The registry maps stable voice ids (what the Android app sends in
``session.create``) to the reference audio/text used for voice cloning.  It is
loaded from ``$ECHOBRIDGE_TTS_VOICE_REGISTRY`` or ``<voice_dir>/voices.json``.

A broken or missing registry must never stop the gateway: the loader keeps
whatever voices are usable and falls back to the legacy single voice configured
through ``ECHOBRIDGE_TTS_VOICE_ID`` / ``ECHOBRIDGE_TTS_REF_AUDIO`` /
``ECHOBRIDGE_TTS_REF_TEXT``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import Settings
from .logging import get_logger

logger = get_logger("echobridge.voices")

VOICE_ID_PATTERN = re.compile(r"^[a-z0-9_]{4,80}$")
DEFAULT_VOICE_ID = "mingyue_v1"

# (candidate number, voice id, display name, description).  The default voice
# stays ``mingyue_v1`` for backward compatibility with saved app settings.
CONFIRMED_VOICES: tuple[tuple[int, str, str, str], ...] = (
    (3, "mingyue_v1", "月月", "温暖自然，默认声线"),
    (1, "mingyue_xixi_v1", "汐汐", "明亮活泼，音调偏高"),
    (2, "mingyue_nuonuo_v1", "糯糯", "软糯温柔，音量偏轻"),
)


@dataclass(frozen=True, slots=True)
class VoiceSpec:
    id: str
    name: str
    description: str
    ref_audio: Path
    ref_text: str
    default: bool = False


@dataclass(frozen=True, slots=True)
class VoiceRegistry:
    voices: tuple[VoiceSpec, ...]
    default_id: str

    def __post_init__(self) -> None:
        if not self.voices:
            raise ValueError("voice registry must contain at least one voice")
        ids = [spec.id for spec in self.voices]
        if len(set(ids)) != len(ids):
            raise ValueError("voice registry contains duplicate ids")
        if self.default_id not in ids:
            raise ValueError("voice registry default_id is not registered")

    def get(self, voice_id: str | None) -> VoiceSpec | None:
        if not isinstance(voice_id, str):
            return None
        target = voice_id.strip()
        if not target:
            return None
        for spec in self.voices:
            if spec.id == target:
                return spec
        return None

    def resolve(self, voice_id: str | None) -> tuple[VoiceSpec, bool]:
        """Return ``(spec, fallback)`` for a requested voice.

        ``None`` and empty strings mean "use the default" without marking the
        session as fallback.  Unknown non-empty ids also use the default but
        set ``fallback=True`` so the app can show a notice.
        """
        spec = self.get(voice_id)
        if spec is not None:
            return spec, False
        default = self.get(self.default_id)
        if default is None:  # __post_init__ guarantees this cannot happen
            raise RuntimeError("voice registry has no default voice")
        if voice_id is None or (isinstance(voice_id, str) and not voice_id.strip()):
            return default, False
        return default, True

    def public(self) -> dict[str, Any]:
        """Wire format for ``GET /voices``; never exposes filesystem paths."""
        return {
            "default": self.default_id,
            "voices": [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "description": spec.description,
                    "default": spec.id == self.default_id,
                }
                for spec in self.voices
            ],
        }


def registry_path(settings: Settings) -> Path:
    raw = (settings.tts_voice_registry or "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    return Path(os.path.expanduser(settings.tts_voice_dir)) / "voices.json"


def load_voice_registry(settings: Settings) -> VoiceRegistry:
    """Load the configured registry, falling back to the legacy single voice."""
    try:
        path = registry_path(settings)
        if path.is_file():
            registry = _load_registry_file(path, settings)
            if registry is not None:
                return registry
            logger.warning("voice registry %s is unusable; using fallback voice", path)
        else:
            logger.info("voice registry %s not found; using fallback voices", path)
        return _auto_registry(settings)
    except Exception:  # noqa: BLE001 - a bad registry must not kill the gateway
        logger.exception("voice registry load failed; using fallback voice")
        return _single_default(settings)


def _load_registry_file(path: Path, settings: Settings) -> VoiceRegistry | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as exc:
        logger.warning("voice registry %s is not readable JSON: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("voice registry %s must contain a JSON object", path)
        return None
    entries = payload.get("voices")
    if not isinstance(entries, list) or not entries:
        logger.warning("voice registry %s has no voices list", path)
        return None

    specs: list[VoiceSpec] = []
    seen: set[str] = set()
    for entry in entries:
        spec = _parse_voice(entry, path)
        if spec is None:
            continue
        if spec.id in seen:
            logger.warning(
                "voice registry %s has duplicate id %s; keeping first", path, spec.id
            )
            continue
        seen.add(spec.id)
        specs.append(spec)
    if not specs:
        logger.warning("voice registry %s has no usable voices", path)
        return None

    raw_default = payload.get("default")
    if not isinstance(raw_default, str) or not raw_default.strip():
        raw_default = settings.tts_voice_id or DEFAULT_VOICE_ID
    default_id = raw_default.strip()
    if default_id not in seen:
        default_id = _fallback_default(settings, specs)

    marked = tuple(
        replace(spec, default=(spec.id == default_id)) for spec in specs
    )
    return VoiceRegistry(marked, default_id)


def _parse_voice(entry: Any, registry_file: Path) -> VoiceSpec | None:
    if not isinstance(entry, dict):
        logger.warning("voice registry %s has a non-object entry", registry_file)
        return None
    raw_id = entry.get("id")
    if not isinstance(raw_id, str) or not VOICE_ID_PATTERN.match(raw_id.strip()):
        logger.warning("voice registry %s has invalid voice id %r", registry_file, raw_id)
        return None
    voice_id = raw_id.strip()

    raw_audio = entry.get("ref_audio")
    if not isinstance(raw_audio, str) or not raw_audio.strip():
        logger.warning("voice %s has no ref_audio", voice_id)
        return None
    ref_audio = _resolve_path(raw_audio, registry_file.parent)
    if not ref_audio.is_file():
        logger.warning("voice %s ref_audio is missing: %s", voice_id, ref_audio)
        return None

    raw_text = entry.get("ref_text")
    if isinstance(raw_text, str) and raw_text.strip():
        ref_text = raw_text.strip()
    else:
        raw_text_file = entry.get("ref_text_file")
        if not isinstance(raw_text_file, str) or not raw_text_file.strip():
            logger.warning("voice %s has no ref_text/ref_text_file", voice_id)
            return None
        ref_text = _read_ref_text(_resolve_path(raw_text_file, registry_file.parent))
        if not ref_text:
            logger.warning(
                "voice %s ref_text_file is empty or missing: %s",
                voice_id,
                raw_text_file,
            )
            return None

    raw_name = entry.get("name")
    name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else voice_id
    raw_description = entry.get("description")
    description = raw_description.strip() if isinstance(raw_description, str) else ""
    return VoiceSpec(
        id=voice_id,
        name=name,
        description=description,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )


def _fallback_default(settings: Settings, specs: list[VoiceSpec]) -> str:
    legacy = _legacy_spec(settings)
    ids = {spec.id for spec in specs}
    if legacy.id in ids:
        logger.warning(
            "voice registry default unavailable; using settings voice %s", legacy.id
        )
        return legacy.id
    if legacy.ref_audio.is_file() and legacy.ref_text:
        logger.warning(
            "voice registry default unavailable; adding settings voice %s", legacy.id
        )
        specs.append(legacy)
        return legacy.id
    logger.warning(
        "voice registry default unavailable; using first usable voice %s", specs[0].id
    )
    return specs[0].id


def _auto_registry(settings: Settings) -> VoiceRegistry:
    voice_dir = Path(os.path.expanduser(settings.tts_voice_dir))
    ref_text = _read_ref_text(voice_dir / "reference.txt")
    if not ref_text:
        ref_text = _read_ref_text(settings.tts_ref_text)
    candidates = [
        voice_dir / f"candidate_{index}.wav"
        for index, _voice_id, _name, _description in CONFIRMED_VOICES
    ]
    if ref_text and all(candidate.is_file() for candidate in candidates):
        specs = tuple(
            VoiceSpec(
                id=voice_id,
                name=name,
                description=description,
                ref_audio=voice_dir / f"candidate_{index}.wav",
                ref_text=ref_text,
                default=(voice_id == DEFAULT_VOICE_ID),
            )
            for index, voice_id, name, description in CONFIRMED_VOICES
        )
        logger.info("auto-registered %d voices from %s", len(specs), voice_dir)
        return VoiceRegistry(specs, DEFAULT_VOICE_ID)
    return _single_default(settings)


def _single_default(settings: Settings) -> VoiceRegistry:
    spec = _legacy_spec(settings)
    return VoiceRegistry((spec,), spec.id)


def _legacy_spec(settings: Settings) -> VoiceSpec:
    voice_id = (settings.tts_voice_id or DEFAULT_VOICE_ID).strip() or DEFAULT_VOICE_ID
    if voice_id == DEFAULT_VOICE_ID:
        name, description = "月月", "温暖自然，默认声线"
    else:
        name, description = voice_id, "默认声线"
    return VoiceSpec(
        id=voice_id,
        name=name,
        description=description,
        ref_audio=_resolve_path(settings.tts_ref_audio, Path.cwd()),
        ref_text=_read_ref_text(settings.tts_ref_text),
        default=True,
    )


def _resolve_path(raw: str, base: Path) -> Path:
    expanded = Path(os.path.expanduser(raw.strip()))
    if expanded.is_absolute():
        return expanded
    return (base / expanded).resolve()


def _read_ref_text(path: str | Path) -> str:
    try:
        candidate = Path(os.path.expanduser(str(path)))
        if not candidate.is_file():
            return ""
        return candidate.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        logger.warning("could not read voice reference text %s: %s", path, exc)
        return ""
