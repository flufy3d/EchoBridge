"""Process-wide GPU arbitration for in-process ASR/TTS.

EchoBridge runs on a single 8 GiB device with one active conversation.  When
ASR and TTS share the process they also share one CUDA context and one caching
allocator, but they must not blindly run at the same time: a TTS decode step can
easily delay an ASR final by several hundred milliseconds.

This module provides a small priority turnstile:

  * ASR acquires with a higher priority than TTS;
  * TTS leases the GPU for one *step* at a time, releasing between steps so a
    waiting ASR request can preempt it;
  * a cancelled coroutine does not release the lease until its worker thread has
    actually finished, otherwise a second inference could overlap it.

The scheduler deliberately has no dependency on torch at import time.  That
keeps the mock/energy test path free of CUDA initialisation.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .logging import get_logger

logger = get_logger("echobridge.gpu")

T = TypeVar("T")

# Priority values are intentionally small and explicit.  A final ASR request
# must beat speculative ASR, which must beat TTS.  The exact numbers only need
# to preserve that ordering.
PRIORITY_ASR_FINAL = 30
PRIORITY_ASR_SPECULATIVE = 20
PRIORITY_TTS = 10
PRIORITY_MAINTENANCE = 0


@dataclass(order=True)
class _Waiter:
    priority: int
    seq: int
    name: str = field(compare=False)
    future: asyncio.Future[None] = field(compare=False)


@dataclass(slots=True)
class GPUMemory:
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    device: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "allocated_mb": round(self.allocated_mb, 1),
            "reserved_mb": round(self.reserved_mb, 1),
            "device": self.device,
        }


class GPUScheduler:
    """Priority-based, process-wide GPU turnstile."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._waiters: list[_Waiter] = []
        self._owner: str | None = None
        self._seq = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lease API
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def lease(
        self,
        name: str,
        *,
        priority: int = PRIORITY_TTS,
        timeout: float | None = None,
    ) -> AsyncIterator[None]:
        """Acquire the GPU turnstile, optionally bounded by ``timeout``.

        A timeout only bounds *waiting for the turnstile*; it never interrupts
        an inference once the lease has been granted.
        """
        waiter = await self._acquire(name, priority, timeout)
        try:
            yield
        finally:
            await self._release(waiter)

    async def _acquire(
        self, name: str, priority: int, timeout: float | None
    ) -> _Waiter:
        with self._lock:
            self._seq += 1
            waiter = _Waiter(
                priority,
                self._seq,
                name,
                asyncio.get_running_loop().create_future(),
            )

        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        async with self._condition:
            self._waiters.append(waiter)
            try:
                while True:
                    self._waiters.sort(key=lambda w: (-w.priority, w.seq))
                    if self._owner is None and self._waiters[0] is waiter:
                        break
                    if deadline is None:
                        await self._condition.wait()
                        continue
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(), timeout=remaining
                        )
                    except asyncio.TimeoutError:
                        raise asyncio.TimeoutError from None
                self._owner = name
                if not waiter.future.done():
                    waiter.future.set_result(None)
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._condition.notify_all()
        return waiter

    async def _release(self, waiter: _Waiter) -> None:
        async with self._condition:
            if self._owner is not None:
                self._owner = None
            self._condition.notify_all()

    @property
    def owner(self) -> str | None:
        return self._owner

    def waiting(self) -> list[str]:
        return [item.name for item in self._waiters]

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    async def run(
        self,
        name: str,
        fn: Callable[..., T],
        *args: Any,
        priority: int = PRIORITY_TTS,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> T:
        """Run blocking ``fn`` in a worker thread while holding the lease.

        Cancellation is safe: the lease is held until the worker thread has
        actually returned.  Without that, cancelling a speculative ASR request
        would let the final request start a second inference on the same GPU
        while the first thread was still inside the model.
        """
        async with self.lease(name, priority=priority, timeout=timeout):
            future = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Let the in-flight kernel/thread finish before releasing.
                try:
                    await asyncio.shield(future)
                except BaseException:
                    pass
                raise

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    @staticmethod
    def memory() -> GPUMemory:
        try:
            import torch

            if not torch.cuda.is_available():
                return GPUMemory()
            device = torch.cuda.current_device()
            return GPUMemory(
                allocated_mb=torch.cuda.memory_allocated(device) / (1024 * 1024),
                reserved_mb=torch.cuda.memory_reserved(device) / (1024 * 1024),
                device=torch.cuda.get_device_name(device),
            )
        except Exception:
            return GPUMemory()

    def status(self, *, include_memory: bool = True) -> dict[str, object]:
        return {
            "owner": self._owner,
            "waiting": self.waiting(),
            # Callers must opt in: touching torch.cuda on a gateway that only
            # talks to an external vLLM worker would create an otherwise
            # unnecessary CUDA context.
            "memory": self.memory().as_dict() if include_memory else {},
        }


_scheduler: GPUScheduler | None = None
_scheduler_lock = threading.Lock()


def get_gpu_scheduler() -> GPUScheduler:
    """Return the process-wide scheduler singleton.

    The singleton is created lazily and without touching CUDA, so importing the
    gateway in tests does not initialise a context.
    """
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = GPUScheduler()
    return _scheduler
