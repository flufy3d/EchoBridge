#!/usr/bin/env python3
"""Dev-only TCP bridge: Windows localhost -> WSL2 gateway.

WSL2 services are not reachable from the Windows Tailscale daemon via
127.0.0.1 on every setup.  This bridge listens on Windows loopback and forwards
bytes to the WSL2 eth0 address, so `tailscale serve --https=8443
http://127.0.0.1:18787` can proxy the gateway to the tailnet.

Production (native Ubuntu) does not use this file.
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except Exception:
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target: str,
    target_port: int,
) -> None:
    try:
        server_reader, server_writer = await asyncio.open_connection(target, target_port)
    except Exception as exc:
        print(f"connect {target}:{target_port} failed: {exc}", flush=True)
        client_writer.close()
        return
    await asyncio.gather(
        _pipe(client_reader, server_writer),
        _pipe(server_reader, client_writer),
    )


def _wsl_ip() -> str:
    output = subprocess.check_output(["wsl", "hostname", "-I"], text=True)
    ips = output.split()
    if not ips:
        raise RuntimeError("wsl hostname -I returned no address")
    return ips[0]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18787)
    parser.add_argument("--target", default="", help="WSL2 IP; auto-detect if omitted")
    parser.add_argument("--target-port", type=int, default=18787)
    args = parser.parse_args()
    target = args.target or _wsl_ip()
    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, target, args.target_port),
        args.listen,
        args.port,
    )
    print(f"proxy {args.listen}:{args.port} -> {target}:{args.target_port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
