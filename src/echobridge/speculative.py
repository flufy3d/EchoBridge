"""Speculative LLM streaming.

The gateway already runs ASR while the VAD endpoint timer is still counting
down.  Once that speculative transcript is ready it can also start the LLM
request; playback still waits for the endpoint, but the provider's first-token
latency is paid in parallel.  If the user resumes speaking, the request is
cancelled and its tokens are wasted.

``SpeculativeLLM`` is a small producer/consumer pair:

* one task drains the real LLM stream into a bounded queue;
* ``deltas()`` replays the queue from the beginning, so the response task never
  issues a second, identical request when the transcript is confirmed;
* cancelling either side unblocks the other.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

from .llm import LLMDelta
from .logging import get_logger

logger = get_logger("echobridge.speculative")


class SpeculativeLLM:
    def __init__(
        self,
        source: Callable[[], AsyncIterator[LLMDelta]],
        *,
        name: str = "",
        queue_size: int = 256,
    ) -> None:
        self._source = source
        self._name = name
        self._queue: asyncio.Queue[LLMDelta | None] = asyncio.Queue(maxsize=queue_size)
        self._task: asyncio.Task[None] | None = None
        self._started = False
        self._error: BaseException | None = None
        self._produced = 0

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    @property
    def error(self) -> BaseException | None:
        return self._error

    @property
    def produced(self) -> int:
        return self._produced

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._task = asyncio.create_task(self._run(), name=f"spec-llm-{self._name}")

    async def _run(self) -> None:
        try:
            async for delta in self._source():
                self._produced += 1
                await self._queue.put(delta)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = exc
            logger.info("speculative llm failed name=%s: %s", self._name, exc)
        finally:
            # Always unblock the consumer.  If it is already gone and the
            # queue is full, dropping the sentinel is harmless.
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def deltas(self) -> AsyncIterator[LLMDelta]:
        """Replay the stream from its first delta.

        The finally block cancels the producer, so a barge-in that stops the
        consumer cannot leave an LLM request running in the background.
        """
        self.start()
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    break
                yield item
        finally:
            self.cancel()

    def cancel(self) -> None:
        task = self._task
        if task is not None and not task.done():
            task.cancel()
