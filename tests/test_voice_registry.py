from __future__ import annotations

import json
from pathlib import Path

from echobridge.config import Settings
from echobridge.voices import (
    CONFIRMED_VOICES,
    VoiceRegistry,
    VoiceSpec,
    load_voice_registry,
)
from tests.helpers import make_settings, write_wav


def _write_registry(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def test_load_registry_resolves_relative_paths_and_public_hides_paths(tmp_path):
    voice_dir = tmp_path / "voices"
    audio = write_wav(voice_dir / "candidate_1.wav")
    text = voice_dir / "reference.txt"
    text.write_text("你好呀，我是溟月。", encoding="utf-8")
    registry_file = _write_registry(
        voice_dir / "voices.json",
        {
            "default": "mingyue_v1",
            "voices": [
                {
                    "id": "mingyue_v1",
                    "name": "月月",
                    "description": "温暖自然，默认声线",
                    "ref_audio": "candidate_1.wav",
                    "ref_text_file": "reference.txt",
                },
                {
                    "id": "mingyue_xixi_v1",
                    "name": "汐汐",
                    "description": "明亮活泼，音调偏高",
                    "ref_audio": str(audio),
                    "ref_text": "内联参考文本",
                },
            ],
        },
    )
    settings = make_settings(
        tts_voice_dir=str(voice_dir), tts_voice_registry=str(registry_file)
    )

    registry = load_voice_registry(settings)

    assert registry.default_id == "mingyue_v1"
    assert [spec.id for spec in registry.voices] == [
        "mingyue_v1",
        "mingyue_xixi_v1",
    ]
    default = registry.get("mingyue_v1")
    assert default is not None
    assert default.ref_audio == audio.resolve()
    assert default.ref_text == "你好呀，我是溟月。"
    assert default.default is True
    assert registry.get("mingyue_xixi_v1").ref_text == "内联参考文本"

    public = registry.public()
    assert public["default"] == "mingyue_v1"
    assert [voice["default"] for voice in public["voices"]] == [True, False]
    encoded = json.dumps(public, ensure_ascii=False)
    assert "ref_audio" not in encoded
    assert "ref_text" not in encoded
    assert str(voice_dir) not in encoded


def test_registry_falls_back_to_legacy_default_when_ref_missing(tmp_path):
    voice_dir = tmp_path / "voices"
    write_wav(voice_dir / "xixi.wav")
    (voice_dir / "reference.txt").write_text("参考文本", encoding="utf-8")
    registry_file = _write_registry(
        voice_dir / "voices.json",
        {
            "default": "mingyue_v1",
            "voices": [
                {
                    "id": "mingyue_v1",
                    "name": "月月",
                    "description": "missing audio",
                    "ref_audio": "missing.wav",
                    "ref_text_file": "reference.txt",
                },
                {
                    "id": "mingyue_xixi_v1",
                    "name": "汐汐",
                    "description": "usable",
                    "ref_audio": "xixi.wav",
                    "ref_text_file": "reference.txt",
                },
            ],
        },
    )
    legacy_audio = write_wav(voice_dir / "legacy.wav")
    legacy_text = voice_dir / "legacy.txt"
    legacy_text.write_text("旧参考文本", encoding="utf-8")
    settings = make_settings(
        tts_voice_dir=str(voice_dir),
        tts_voice_registry=str(registry_file),
        tts_voice_id="mingyue_v1",
        tts_ref_audio=str(legacy_audio),
        tts_ref_text=str(legacy_text),
    )

    registry = load_voice_registry(settings)

    assert registry.default_id == "mingyue_v1"
    assert {spec.id for spec in registry.voices} == {
        "mingyue_v1",
        "mingyue_xixi_v1",
    }
    default = registry.get("mingyue_v1")
    assert default is not None
    assert default.ref_audio == legacy_audio
    assert default.ref_text == "旧参考文本"


def test_bad_registry_json_falls_back_without_raising(tmp_path):
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    registry_file = voice_dir / "voices.json"
    registry_file.write_text("{not valid json", encoding="utf-8")
    legacy_audio = write_wav(voice_dir / "legacy.wav")
    legacy_text = voice_dir / "legacy.txt"
    legacy_text.write_text("旧参考文本", encoding="utf-8")
    settings = make_settings(
        tts_voice_dir=str(voice_dir),
        tts_voice_registry=str(registry_file),
        tts_voice_id="mingyue_v1",
        tts_ref_audio=str(legacy_audio),
        tts_ref_text=str(legacy_text),
    )

    registry = load_voice_registry(settings)

    assert registry.default_id == "mingyue_v1"
    assert len(registry.voices) == 1
    assert registry.voices[0].ref_audio == legacy_audio
    assert registry.voices[0].default is True


def test_auto_registers_three_confirmed_candidates(tmp_path):
    voice_dir = tmp_path / "voices"
    for index, _voice_id, _name, _description in CONFIRMED_VOICES:
        write_wav(voice_dir / f"candidate_{index}.wav")
    (voice_dir / "reference.txt").write_text("确认参考文本", encoding="utf-8")
    settings = make_settings(
        tts_voice_dir=str(voice_dir),
        tts_voice_registry="",
        tts_ref_audio=str(voice_dir / "reference.wav"),
        tts_ref_text=str(voice_dir / "reference.txt"),
    )

    registry = load_voice_registry(settings)

    assert registry.default_id == "mingyue_v1"
    assert [spec.id for spec in registry.voices] == [
        "mingyue_v1",
        "mingyue_xixi_v1",
        "mingyue_nuonuo_v1",
    ]
    assert registry.get("mingyue_v1").ref_audio == voice_dir / "candidate_3.wav"
    assert registry.get("mingyue_xixi_v1").name == "汐汐"
    assert registry.get("mingyue_nuonuo_v1").name == "糯糯"


def test_auto_registration_requires_all_three_candidates(tmp_path):
    voice_dir = tmp_path / "voices"
    write_wav(voice_dir / "candidate_3.wav")
    (voice_dir / "reference.txt").write_text("参考文本", encoding="utf-8")
    settings = make_settings(
        tts_voice_dir=str(voice_dir),
        tts_voice_registry="",
        tts_ref_audio=str(voice_dir / "reference.wav"),
        tts_ref_text=str(voice_dir / "reference.txt"),
    )

    registry = load_voice_registry(settings)

    assert len(registry.voices) == 1
    assert registry.default_id == "mingyue_v1"
    assert registry.voices[0].ref_audio == voice_dir / "reference.wav"


def test_invalid_and_duplicate_voices_are_dropped(tmp_path):
    voice_dir = tmp_path / "voices"
    audio = write_wav(voice_dir / "voice.wav")
    text = voice_dir / "reference.txt"
    text.write_text("参考文本", encoding="utf-8")
    registry_file = _write_registry(
        voice_dir / "voices.json",
        {
            "default": "mingyue_v1",
            "voices": [
                {
                    "id": "mingyue_v1",
                    "name": "月月",
                    "ref_audio": "voice.wav",
                    "ref_text_file": "reference.txt",
                },
                {
                    "id": "mingyue_v1",
                    "name": "duplicate",
                    "ref_audio": str(audio),
                    "ref_text_file": "reference.txt",
                },
                {
                    "id": "Bad-ID",
                    "name": "invalid id",
                    "ref_audio": str(audio),
                    "ref_text_file": "reference.txt",
                },
                {
                    "id": "missing_audio_v1",
                    "name": "missing",
                    "ref_audio": "nope.wav",
                    "ref_text_file": "reference.txt",
                },
            ],
        },
    )
    settings = make_settings(
        tts_voice_dir=str(voice_dir), tts_voice_registry=str(registry_file)
    )

    registry = load_voice_registry(settings)

    assert [spec.id for spec in registry.voices] == ["mingyue_v1"]
    assert registry.get("mingyue_v1").name == "月月"


def test_resolve_empty_and_unknown_voice(tmp_path):
    registry = VoiceRegistry(
        (
            # Minimal direct construction; no files are read by resolve().
            VoiceSpec(
                id="mingyue_v1",
                name="月月",
                description="",
                ref_audio=tmp_path / "a.wav",
                ref_text="x",
                default=True,
            ),
            VoiceSpec(
                id="mingyue_xixi_v1",
                name="汐汐",
                description="",
                ref_audio=tmp_path / "b.wav",
                ref_text="x",
            ),
        ),
        "mingyue_v1",
    )

    spec, fallback = registry.resolve(None)
    assert (spec.id, fallback) == ("mingyue_v1", False)
    spec, fallback = registry.resolve("")
    assert (spec.id, fallback) == ("mingyue_v1", False)
    spec, fallback = registry.resolve("  ")
    assert (spec.id, fallback) == ("mingyue_v1", False)
    spec, fallback = registry.resolve("unknown_voice")
    assert (spec.id, fallback) == ("mingyue_v1", True)
    spec, fallback = registry.resolve("mingyue_xixi_v1")
    assert (spec.id, fallback) == ("mingyue_xixi_v1", False)


def test_settings_read_voice_paths_from_env(monkeypatch):
    monkeypatch.setenv("ECHOBRIDGE_DEV", "1")
    monkeypatch.setenv("ECHOBRIDGE_TTS_VOICE_DIR", "/tmp/voices")
    monkeypatch.setenv("ECHOBRIDGE_TTS_VOICE_REGISTRY", "/tmp/voices/custom.json")
    settings = Settings.from_env()
    assert settings.tts_voice_dir == "/tmp/voices"
    assert settings.tts_voice_registry == "/tmp/voices/custom.json"
