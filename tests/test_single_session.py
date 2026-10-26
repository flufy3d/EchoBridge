from __future__ import annotations

import asyncio
import json

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from tests.helpers import make_settings, session_create
from tests.test_ws_integration import start_server, stop_server


async def connect(port, device_id, token):
    return await websockets.connect(
        f"ws://127.0.0.1:{port}/echobridge/ws?device_id={device_id}",
        additional_headers={
            "Authorization": f"Bearer {token}",
            "X-Device-Id": device_id,
        },
        subprotocols=["echobridge"],
        open_timeout=3,
    )


async def recv_type(ws, event_type, timeout=3.0):
    async def wait():
        while True:
            event = json.loads(await ws.recv())
            if event.get("type") == event_type:
                return event

    return await asyncio.wait_for(wait(), timeout)


async def test_same_device_takeover_and_other_device_rejection():
    settings = make_settings()
    server, task, port = await start_server(settings)
    try:
        first = await connect(port, "x10a", settings.token)
        await first.send(json.dumps(session_create()))
        await recv_type(first, "session.created")

        second = await connect(port, "x10a", settings.token)
        await second.send(json.dumps(session_create()))
        await recv_type(second, "session.created")
        # The old socket is closed with a clean session.closed first.
        closed = await recv_type(first, "session.closed")
        assert closed["reason"] == "superseded"
        await first.close()

        third = await connect(port, "other-device", settings.token)
        with pytest.raises(ConnectionClosed) as exc:
            await third.recv()
        assert exc.value.rcvd is not None
        assert exc.value.rcvd.code == 1013
        await second.send(json.dumps({"type": "session.close"}))
        await recv_type(second, "session.closed")
        await second.close()
    finally:
        await stop_server(server, task)
