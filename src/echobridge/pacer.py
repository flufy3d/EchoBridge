"""20 ms audio pacer for downstream PCM.

The pacer is the only component that writes `response.output_audio.*` events.
It guarantees:
  * started is emitted before the first delta;
  * frames are sent on a 20 ms cadence;
  * a bounded queue applies backpressure;
  * cancelled/stale generations never produce audio after `response.canceled`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from . import protocol
from .audio import OUTPUT_FRAME_BYTES, pcm16_b64
from .logging import get_logger

logger = get_logger("echobridge.pacer")

SendEvent = Callable[[dict], Awaitable[bool]]


class AudioPacer:
    def __init__(
        self,
        send_event: SendEvent,
        ctx: protocol.ResponseContext,
        interval: float = 0.020,
        queue_size: int = 200,
    ) -> None:
        self._send_event = send_event
        self._ctx = ctx
        self._interval = interval
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self.started = False
        self.done = False

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name=f"pacer-{self._ctx.response_id}")

    async def feed(self, pcm: bytes) -> None:
        if pcm:
            await self._queue.put(bytes(pcm))

    async def finish(self) -> None:
        await self._queue.put(None)
        if self._task is not None:
            await self._task

    async def wait(self) -> None:
        if self._task is not None:
            await self._task

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        pending = b""

        async def emit(frame: bytes) -> bool:
            nonlocal deadline
            if not self.started:
                ok = await self._send_event(protocol.audio_started(self._ctx), self._ctx)
                if not ok:
                    return False
                self.started = True
            ok = await self._send_event(
                protocol.audio_delta(self._ctx, pcm16_b64(frame)), self._ctx
            )
            if not ok:
                return False
            deadline += self._interval
            now = loop.time()
            if deadline > now:
                await asyncio.sleep(deadline - now)
            else:
                # We fell behind (GC, slow client, scheduler).  Do not
                # accumulate debt; restart the clock from now.
                deadline = now
            return True

        try:
            while True:
                chunk = await self._queue.get()
                if chunk is None:
                    break
                # TTS/dispatcher reads are not frame-aligned (64 KiB/30 KiB is
                # typical).  Carry the tail into the next read instead of
                # padding every read to 20 ms: the old path inserted 15-20 ms
                # of silence every ~2 s and was audible as a click.
                pending += chunk
                usable = len(pending) - (len(pending) % OUTPUT_FRAME_BYTES)
                if not usable:
                    continue
                block = pending[:usable]
                pending = pending[usable:]
                for offset in range(0, usable, OUTPUT_FRAME_BYTES):
                    if not await emit(block[offset : offset + OUTPUT_FRAME_BYTES]):
                        return
            if pending:
                # End of response: the only place padding is allowed.
                padded = pending + b"\x00" * (OUTPUT_FRAME_BYTES - len(pending))
                if not await emit(padded):
                    return
            if self.started:
                await self._send_event(protocol.audio_done(self._ctx), self._ctx)
            self.done = True
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("pacer failed response=%s", self._ctx.response_id)
            raise
