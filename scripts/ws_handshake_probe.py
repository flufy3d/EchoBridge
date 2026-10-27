#!/usr/bin/env python3
"""Probe an EchoBridge WebSocket handshake without printing the token.

Examples:
  # Through Windows Tailscale Serve (default URL)
  python scripts/ws_handshake_probe.py --auth bearer --expect 101

  # Verify a wrong token is rejected
  python scripts/ws_handshake_probe.py --auth bearer --token wrong --expect 403

  # Verify the browser-probe subprotocol path
  python scripts/ws_handshake_probe.py --auth subprotocol --expect 101

  # Direct gateway, no TLS
  python scripts/ws_handshake_probe.py \\
      --url ws://127.0.0.1:18787/echobridge/ws --auth bearer --expect 101

The script prints only the auth source, token length, HTTP status and the
`session.created` id.  It never prints Authorization, the token, or the
Sec-WebSocket-Protocol header value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import websockets
from websockets.exceptions import InvalidStatus

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = "wss://work-pc.tailea818a.ts.net:8443/echobridge/ws"
DEFAULT_TOKEN_FILE = ROOT / "var" / "dev-token"


def load_token(args: argparse.Namespace) -> tuple[str, str]:
    if args.token is not None:
        return args.token.strip(), "arg"
    if args.token_file:
        path = Path(args.token_file)
        return path.read_text(encoding="utf-8").strip(), "file"
    env = os.environ.get("ECHOBRIDGE_TOKEN", "").strip()
    if env:
        return env, "env"
    if DEFAULT_TOKEN_FILE.exists():
        return DEFAULT_TOKEN_FILE.read_text(encoding="utf-8").strip(), "file"
    return "", "none"


def session_create() -> dict:
    return {
        "type": "session.create",
        "event_id": "probe_create",
        "session": {
            "model": "echobridge-local-v1",
            "instructions": "",
            "audio": {
                "input": {"format": {"type": "pcm", "rate": 16000}},
                "output": {
                    "format": {"type": "pcm_s16le", "rate": 24000},
                    "voice": "mingyue_v1",
                },
            },
            "tools": [],
        },
    }


async def run(args: argparse.Namespace) -> int:
    token, source = ("", "none") if args.auth == "none" else load_token(args)
    if args.auth != "none" and not token:
        print(
            "token missing: use --token, --token-file, ECHOBRIDGE_TOKEN, "
            "or var/dev-token",
            file=sys.stderr,
        )
        return 2

    headers = {"X-Device-Id": args.device_id}
    protocols: list[str] | None = None
    if args.auth in ("bearer", "both"):
        headers["Authorization"] = f"Bearer {token}"
    if args.auth in ("subprotocol", "both"):
        protocols = ["echobridge", f"token.{token}"]

    print(f"url={args.url}")
    print(f"auth={args.auth} token_source={source} token_len={len(token)}")
    try:
        async with websockets.connect(
            args.url,
            additional_headers=headers,
            subprotocols=protocols,
            origin=args.origin or None,
            open_timeout=args.timeout,
            close_timeout=2,
        ) as ws:
            print(f"result=101 subprotocol={ws.subprotocol!r}")
            if args.session_create:
                await ws.send(json.dumps(session_create()))
                raw = await asyncio.wait_for(ws.recv(), timeout=args.timeout)
                event = json.loads(raw)
                if event.get("type") != "session.created":
                    print(f"unexpected_event={event.get('type')!r}")
                    return 1
                session = event.get("session") or {}
                print(f"session.created id={session.get('id', '')}")
                await ws.send(json.dumps({"type": "session.close"}))
                try:
                    await asyncio.wait_for(ws.recv(), timeout=2)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    pass
            if args.expect not in ("101", "any"):
                print(f"expected={args.expect} but handshake succeeded")
                return 1
            return 0
    except InvalidStatus as exc:
        status = exc.response.status_code
        reason = exc.response.reason_phrase
        print(f"result={status} {reason}")
        if status in (401, 403):
            print(
                "hint=gateway rejected the presented credential; compare "
                "token_source with the running gateway's ECHOBRIDGE_TOKEN "
                "(dev.sh uses var/dev-token)"
            )
        if args.expect in (str(status), "any"):
            return 0
        print(f"expected={args.expect} but got {status}")
        return 1
    except Exception as exc:  # noqa: BLE001 - probe should report any failure
        print(f"error={type(exc).__name__}: {exc}")
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="WebSocket URL")
    parser.add_argument(
        "--auth",
        choices=["bearer", "subprotocol", "both", "none"],
        default="bearer",
        help="which credential transport to send",
    )
    parser.add_argument("--token", default=None, help="literal token (testing only)")
    parser.add_argument("--token-file", default="", help="token file path")
    parser.add_argument("--device-id", default="ws-probe")
    parser.add_argument("--origin", default="", help="optional Origin header")
    parser.add_argument(
        "--expect",
        choices=["101", "401", "403", "any"],
        default="101",
        help="expected HTTP status; 101 means a successful upgrade",
    )
    parser.add_argument(
        "--no-session-create",
        dest="session_create",
        action="store_false",
        help="stop after the handshake",
    )
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
