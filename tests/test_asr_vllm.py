from __future__ import annotations

import httpx
import pytest

from echobridge.asr import ASRError, ASRUnavailableError, VLLMASRClient
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


async def _client_with(handler) -> VLLMASRClient:
    settings = make_settings(asr_backend="vllm", asr_url="http://asr.test")
    client = VLLMASRClient(settings)
    await client._client.aclose()
    # The test owns a transport-backed client; close it in the test.
    client._client = httpx.AsyncClient(
        base_url="http://asr.test", transport=httpx.MockTransport(handler)
    )
    return client


async def test_vllm_asr_connect_error_is_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = await _client_with(handler)
    try:
        with pytest.raises(ASRUnavailableError):
            await client.transcribe(b"\x00\x00" * 16_000, 16_000)
    finally:
        await client.close()

    assert client.asr_health.available is False
    assert client.asr_health.unavailable_total == 1
    assert client.asr_health.last_error == "ConnectError"


async def test_vllm_asr_500_is_unavailable():
    client = await _client_with(lambda request: httpx.Response(500))
    try:
        with pytest.raises(ASRUnavailableError):
            await client.transcribe(b"\x00\x00" * 16_000, 16_000)
    finally:
        await client.close()

    assert client.asr_health.available is False
    assert client.asr_health.last_error == "HTTP 500"


async def test_vllm_asr_400_is_non_retryable():
    client = await _client_with(lambda request: httpx.Response(400))
    try:
        with pytest.raises(ASRError):
            await client.transcribe(b"\x00\x00" * 16_000, 16_000)
    finally:
        await client.close()

    assert client.asr_health.available is True
    assert client.asr_health.error_total == 1


async def test_vllm_asr_empty_200_is_healthy():
    client = await _client_with(lambda request: httpx.Response(200, json={"text": ""}))
    try:
        result = await client.transcribe(b"\x00\x00" * 16_000, 16_000)
    finally:
        await client.close()

    assert result.text == ""
    assert client.asr_health.available is True
    assert client.asr_health.last_ok > 0
