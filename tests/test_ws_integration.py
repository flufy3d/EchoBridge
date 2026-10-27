from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
import uvicorn
import websockets
from websockets.exceptions import InvalidStatus

from echobridge.app import create_app
from tests.helpers import make_settings, session_create


async def start_server(settings):
    app = create_app(settings)
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="on",
        access_log=False,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


async def stop_server(server, task):
    server.should_exit = True
    await asyncio.wait_for(task, timeout=5)


async def recv_until(ws, event_type: str, timeout: float = 5.0):
    async def receive():
        while True:
            raw = await ws.recv()
            event = json.loads(raw)
            if event.get("type") == event_type:
                return event

    return await asyncio.wait_for(receive(), timeout=timeout)


async def test_health_and_probe():
    settings = make_settings()
    server, task, port = await start_server(settings)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
            health = await client.get("/healthz")
            assert health.status_code == 200
            payload = health.json()
            assert payload["ok"] is True
            assert payload["service"] == "echobridge-gateway"
            assert payload["uptime_s"] >= 0
            assert "backend" not in payload
            ready = await client.get("/readyz")
            assert ready.status_code == 200
            assert ready.json()["ok"] is True
            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert "echobridge_up 1" in metrics.text
            assert "echobridge_active_session 0" in metrics.text
            probe = await client.get("/probe")
            assert probe.status_code == 200
            assert "EchoBridge Phase 1 Probe" in probe.text
    finally:
        await stop_server(server, task)


@pytest.mark.parametrize(
    ("headers", "subprotocols", "expected"),
    [
        ({}, None, 403),
        ({"Authorization": "Bearer wrong"}, None, 403),
        ({"Authorization": "Bearer test-token"}, None, 101),
        ({}, ["echobridge", "token.test-token"], 101),
        (
            {"Authorization": "Bearer test-token"},
            ["echobridge", "token.test-token"],
            101,
        ),
    ],
)
async def test_websocket_auth_matrix(headers, subprotocols, expected):
    settings = make_settings()
    server, task, port = await start_server(settings)
    try:
        if expected == 403:
            with pytest.raises(InvalidStatus) as exc:
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}/echobridge/ws",
                    additional_headers=headers,
                    subprotocols=subprotocols,
                    open_timeout=3,
                ):
                    pass
            assert exc.value.response.status_code == 403
            return

        async with websockets.connect(
            f"ws://127.0.0.1:{port}/echobridge/ws",
            additional_headers=headers,
            subprotocols=subprotocols,
            open_timeout=3,
        ) as ws:
            await ws.send(json.dumps(session_create(), ensure_ascii=False))
            created = await recv_until(ws, "session.created")
            assert created["session"]["id"]
            await ws.send(json.dumps({"type": "session.close"}))
            closed = await recv_until(ws, "session.closed")
            assert closed["reason"] == "client"
    finally:
        await stop_server(server, task)


async def test_websocket_auth_log_is_redacted(caplog):
    settings = make_settings(token="super-secret-token-value")
    wrong = "wrong-secret"
    server, task, port = await start_server(settings)
    try:
        with caplog.at_level(logging.WARNING, logger="echobridge.app"):
            with pytest.raises(InvalidStatus):
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}/echobridge/ws",
                    additional_headers={"Authorization": f"Bearer {wrong}"},
                    open_timeout=3,
                ):
                    pass

        text = caplog.text
        assert "auth=bearer" in text
        assert f"auth_len={len('Bearer ' + wrong)}" in text
        assert "subprotocol_len=0" in text
        assert "tailscale_user=no" in text
        assert settings.token not in text
        assert wrong not in text

        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="echobridge.app"):
            with pytest.raises(InvalidStatus):
                async with websockets.connect(
                    f"ws://127.0.0.1:{port}/echobridge/ws",
                    open_timeout=3,
                ):
                    pass

        text = caplog.text
        assert "auth=none" in text
        assert "auth_len=0" in text
        assert "subprotocol_len=0" in text
    finally:
        await stop_server(server, task)


async def test_websocket_accept_log_is_redacted(caplog):
    settings = make_settings(token="super-secret-token-value")
    server, task, port = await start_server(settings)
    try:
        with caplog.at_level(logging.INFO, logger="echobridge.app"):
            async with websockets.connect(
                f"ws://127.0.0.1:{port}/echobridge/ws",
                additional_headers={
                    "Authorization": f"Bearer {settings.token}",
                    "Origin": "https://example.invalid",
                    "Tailscale-User-Login": "user@example.invalid",
                    "X-Forwarded-For": "100.64.0.1",
                },
                open_timeout=3,
            ) as ws:
                await ws.send(json.dumps(session_create(), ensure_ascii=False))
                await recv_until(ws, "session.created")
                await ws.send(json.dumps({"type": "session.close"}))
                await recv_until(ws, "session.closed")

        text = caplog.text
        assert "accepted websocket" in text
        assert "auth=bearer" in text
        assert "origin=yes" in text
        assert "tailscale_user=yes" in text
        assert "xff=yes" in text
        assert settings.token not in text
        assert "user@example.invalid" not in text
    finally:
        await stop_server(server, task)


async def test_websocket_greeting_turn_cancel_close():
    settings = make_settings(tts_tone_char_ms=6, tts_first_chunk_ms=5)
    server, task, port = await start_server(settings)
    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{port}/echobridge/ws?device_id=integration&asr_text="
            "%E4%BD%A0%E5%A5%BD%E5%91%80%EF%BC%8C%E6%BA%9F%E6%9C%88",
            additional_headers={
                "Authorization": f"Bearer {settings.token}",
                "X-Device-Id": "integration",
            },
            subprotocols=["echobridge"],
            open_timeout=3,
        ) as ws:
            await ws.send(json.dumps(session_create(), ensure_ascii=False))
            created = await recv_until(ws, "session.created")
            assert created["session"]["id"]
            await ws.send(
                json.dumps(
                    {
                        "type": "speech_text_buffer.append",
                        "speech_id": "g1",
                        "text": "你来啦，",
                    },
                    ensure_ascii=False,
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "speech_text_buffer.commit",
                        "speech_id": "g1",
                        "text": "我在呢。",
                    },
                    ensure_ascii=False,
                )
            )
            started = await recv_until(ws, "response.output_audio.started")
            assert started["response_id"]
            await recv_until(ws, "response.output_audio.delta")
            await recv_until(ws, "response.output_audio.done")
            await recv_until(ws, "response.done")

            # A short simulated speech segment triggers the mock ASR.
            import base64

            from tests.helpers import loud_frame, silence_frame

            for _ in range(30):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(loud_frame()).decode("ascii"),
                        }
                    )
                )
            for _ in range(35):
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(silence_frame()).decode("ascii"),
                        }
                    )
                )
            completed = await recv_until(ws, "conversation.item.input_audio_transcription.completed")
            assert completed["transcript"]
            await recv_until(ws, "response.output_text.delta")
            await recv_until(ws, "response.done")

            await ws.send(json.dumps({"type": "session.close"}))
            closed = await recv_until(ws, "session.closed")
            assert closed["reason"] == "client"
    finally:
        await stop_server(server, task)
