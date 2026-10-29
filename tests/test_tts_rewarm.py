from __future__ import annotations

from echobridge.tts import GREETING, VLLMTTSClient
from tests.helpers import make_settings, make_voice_registry


async def test_vllm_rewarm_forces_a_request_despite_the_greeting_cache(tmp_path):
    registry = make_voice_registry(tmp_path)
    settings = make_settings(
        tts_backend="vllm",
        tts_url="http://tts.test",
        backchannel_enabled=False,
    )
    client = VLLMTTSClient(settings, registry)
    calls: list[str] = []

    async def fake_stream(text: str, voice: str):
        calls.append(text)
        yield b"\x00\x00" * 2_400

    client._stream = fake_stream
    try:
        async for _ in client.synthesize(GREETING, registry.default_id, 1):
            pass
        assert calls == [GREETING]

        # The normal path is now a cache hit and must not touch the worker.
        async for _ in client.synthesize(GREETING, registry.default_id, 1):
            pass
        assert calls == [GREETING]

        # A restarted worker has an empty model cache; rewarm must issue a real
        # request even though the gateway still has the old PCM.
        await client.rewarm(registry.default_id)
        assert calls == [GREETING, GREETING]
    finally:
        await client.close()
