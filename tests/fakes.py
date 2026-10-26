from __future__ import annotations

import asyncio
import json
from typing import Any


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise ConnectionError("closed")
        return item

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason
        await self.incoming.put(None)

    async def push(self, event: dict[str, Any]) -> None:
        await self.incoming.put(json.dumps(event, ensure_ascii=False))

    def types(self) -> list[str]:
        return [str(item.get("type")) for item in self.sent]

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [item for item in self.sent if item.get("type") == event_type]

    async def wait_for_type(self, event_type: str, timeout: float = 3.0) -> dict[str, Any]:
        async def wait() -> dict[str, Any]:
            while True:
                for item in self.sent:
                    if item.get("type") == event_type:
                        return item
                await asyncio.sleep(0.005)

        return await asyncio.wait_for(wait(), timeout=timeout)

    async def wait_until(self, predicate, timeout: float = 3.0) -> None:
        async def wait() -> None:
            while not predicate():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait(), timeout=timeout)
