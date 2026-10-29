from __future__ import annotations

import base64
import json

from echobridge.tts import MockTTS, VLLMTTSClient
from tests.helpers import make_settings, make_voice_registry, write_wav


async def _collect(tts, text: str, voice: str) -> bytes:
    chunks = []
    async for chunk in tts.synthesize(text, voice, 1):
        chunks.append(chunk)
    return b"".join(chunks)


def _ref_bytes(payload: dict) -> bytes:
    prefix = "data:audio/wav;base64,"
    assert payload["ref_audio"].startswith(prefix)
    return base64.b64decode(payload["ref_audio"][len(prefix) :])


async def test_vllm_payload_uses_per_voice_reference(tmp_path):
    registry = make_voice_registry(tmp_path)
    settings = make_settings(
        tts_backend="vllm",
        tts_url="http://127.0.0.1:9",
        tts_voice_dir=str(tmp_path),
    )
    client = VLLMTTSClient(settings, registry)
    try:
        xixi = registry.get("mingyue_xixi_v1")
        nuonuo = registry.get("mingyue_nuonuo_v1")
        default = registry.get(registry.default_id)
        assert xixi is not None and nuonuo is not None and default is not None

        payload = client._payload("你好呀", "mingyue_xixi_v1")
        assert payload["input"] == "你好呀"
        assert payload["ref_text"] == xixi.ref_text
        assert _ref_bytes(payload) == xixi.ref_audio.read_bytes()

        payload = client._payload("你好呀", "mingyue_nuonuo_v1")
        assert payload["ref_text"] == nuonuo.ref_text
        assert _ref_bytes(payload) == nuonuo.ref_audio.read_bytes()

        # Unknown and empty voices must use the default reference, not a ghost.
        for voice in ("", "unknown_voice"):
            payload = client._payload("你好呀", voice)
            assert payload["ref_text"] == default.ref_text
            assert _ref_bytes(payload) == default.ref_audio.read_bytes()
    finally:
        await client.close()


async def test_mock_tts_produces_distinct_tones_per_voice(tmp_path):
    registry = make_voice_registry(tmp_path)
    settings = make_settings(
        tts_backend="mock",
        tts_first_chunk_ms=0,
        tts_tone_char_ms=4,
    )
    tts = MockTTS(settings, registry)
    try:
        default = await _collect(tts, "你好呀，溟月", registry.default_id)
        xixi = await _collect(tts, "你好呀，溟月", "mingyue_xixi_v1")
        nuonuo = await _collect(tts, "你好呀，溟月", "mingyue_nuonuo_v1")
        assert default and xixi and nuonuo
        assert len({default, xixi, nuonuo}) == 3

        # Unknown voice resolves to the default tone; cached() uses the same
        # resolution so backchannel audio cannot leak another voice's cache.
        assert await _collect(tts, "你好呀，溟月", "unknown_voice") == default
        assert tts.cached("嗯～", "unknown_voice") == tts.cached(
            "嗯～", registry.default_id
        )
    finally:
        await tts.close()


async def test_vllm_cached_uses_default_for_empty_voice(tmp_path):
    registry = make_voice_registry(tmp_path)
    settings = make_settings(
        tts_backend="vllm",
        tts_url="http://127.0.0.1:9",
        tts_voice_dir=str(tmp_path),
    )
    client = VLLMTTSClient(settings, registry)
    try:
        assert client.cached("你好", "") is None
        assert client.cached("你好", "unknown_voice") is None
    finally:
        await client.close()


async def test_backchannel_clips_are_pinned_without_synthesis(tmp_path):
    registry = make_voice_registry(tmp_path)
    clip_dir = tmp_path / "backchannel" / "mingyue_v1"
    write_wav(clip_dir / "think.wav", sample_rate=24_000, duration_ms=400)
    (clip_dir / "clips.json").write_text(
        json.dumps({"我想想～": "think.wav", "unused": "missing.wav"}),
        encoding="utf-8",
    )
    settings = make_settings(
        tts_backend="vllm",
        tts_url="http://127.0.0.1:9",
        tts_voice_dir=str(tmp_path),
        backchannel_phrases=("我想想～", "这个嘛～"),
    )
    client = VLLMTTSClient(settings, registry)
    try:
        clip = client.cached("我想想～", "mingyue_v1")
        assert clip is not None and len(clip) > 0
        # Phrases without a clip, and other voices, still go through TTS.
        assert client.cached("这个嘛～", "mingyue_v1") is None
        assert client.cached("我想想～", "mingyue_xixi_v1") is None

        synthesized: list[str] = []

        async def fake_prefetch(text: str, voice: str) -> bytes:
            synthesized.append(text)
            return b"\x00\x01" * 100

        client._prefetch = fake_prefetch  # type: ignore[method-assign]
        await client.ensure_warm("mingyue_v1")
        assert "我想想～" not in synthesized
        assert "这个嘛～" in synthesized
        assert client.cached("我想想～", "mingyue_v1") == clip
    finally:
        await client.close()


async def test_backchannel_clips_ignored_when_disabled(tmp_path):
    registry = make_voice_registry(tmp_path)
    clip_dir = tmp_path / "backchannel" / "mingyue_v1"
    write_wav(clip_dir / "think.wav", sample_rate=24_000, duration_ms=400)
    (clip_dir / "clips.json").write_text(
        json.dumps({"我想想～": "think.wav"}), encoding="utf-8"
    )
    settings = make_settings(
        tts_backend="vllm",
        tts_url="http://127.0.0.1:9",
        tts_voice_dir=str(tmp_path),
        backchannel_enabled=False,
        backchannel_phrases=("我想想～",),
    )
    client = VLLMTTSClient(settings, registry)
    try:
        assert client.cached("我想想～", "mingyue_v1") is None
    finally:
        await client.close()

