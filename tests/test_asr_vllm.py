from __future__ import annotations

import httpx

from echobridge.asr import VLLMASRClient
from tests.helpers import make_settings


async def test_vllm_asr_uploads_wav_with_context():
    settings = make_settings(
        asr_backend="vllm",
        asr_url="http://asr.test",
        asr_hotwords=("溟月",),
    )
    client = VLLMASRClient(settings)
    seen: dict[str, bool] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        seen["path"] = request.url.path == "/v1/audio/transcriptions"
        seen["wav"] = b"RIFF" in body and b"WAVE" in body
        seen["context"] = "专有名词".encode("utf-8") in body
        seen["language"] = b"\r\nzh\r\n" in body
        return httpx.Response(200, json={"text": "你好呀，溟月"})

    await client._client.aclose()
    client._client = httpx.AsyncClient(
        base_url="http://asr.test", transport=httpx.MockTransport(handler)
    )
    try:
        result = await client.transcribe(b"\x00\x00" * 16_000, 16_000)
    finally:
        await client.close()

    assert result.text == "你好呀，溟月"
    assert result.language == "zh"
    assert all(seen.values()), seen
