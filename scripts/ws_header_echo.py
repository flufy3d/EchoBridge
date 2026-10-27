#!/usr/bin/env python3
"""Minimal WebSocket server that reports header names/presence/lengths.

Use it behind Tailscale Serve to prove whether Authorization and
Sec-WebSocket-Protocol survive the proxy:

  python scripts/ws_header_echo.py --port 19443 --once
  tailscale serve --bg --https=9443 http://127.0.0.1:19443
  python scripts/ws_handshake_probe.py \\
      --url wss://work-pc.tailea818a.ts.net:9443/ \\
      --auth both --no-session-create --expect 101

The report deliberately contains no credential values.  Token-shaped
subprotocol items are replaced with ``token.<redacted>``.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets


def report_for(ws: websockets.WebSocketServerProtocol) -> dict:
    headers = ws.request.headers
    auth = headers.get("Authorization", "")
    protocols = headers.get("Sec-WebSocket-Protocol", "")
    items = [item.strip() for item in protocols.split(",") if item.strip()]
    redacted = [
        "token.<redacted>" if item.startswith("token.") else item for item in items
    ]
    return {
        "header_names": sorted(headers.keys()),
        "authorization_present": bool(auth),
        "authorization_len": len(auth),
        "authorization_scheme": auth.split(" ", 1)[0] if auth else "",
        "subprotocol_len": len(protocols),
        "subprotocol_items": redacted,
        "origin_present": bool(headers.get("Origin")),
        "tailscale_user_present": bool(headers.get("Tailscale-User-Login")),
        "x_forwarded_for_present": bool(headers.get("X-Forwarded-For")),
    }


async def run(args: argparse.Namespace) -> None:
    stop = asyncio.Event()

    async def handler(ws):
        report = report_for(ws)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        await ws.send(json.dumps(report, ensure_ascii=False))
        await ws.close()
        if args.once:
            stop.set()

    async with websockets.serve(handler, args.host, args.port):
        print(
            f"listening ws://{args.host}:{args.port}/ "
            f"(Tailscale Serve target http://127.0.0.1:{args.port})",
            flush=True,
        )
        await stop.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19443)
    parser.add_argument("--once", action="store_true", help="exit after one connection")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
