"""Idle keepalive for the local TTS worker.

A long-idle Qwen3-TTS worker on WSL2 can find its CUDA context in a bad state
when the first real request arrives hours later.  Synthesising one very short
sentence every ``ECHOBRIDGE_TTS_KEEPALIVE_S`` seconds keeps the worker warm and
turns a latent failure into an early, logged probe.  The PCM is discarded.

The probe never runs while a response is in flight and never touches the
session's first-audio/backchannel metrics: it has no ``ResponseContext`` and
does not go through :class:`~echobridge.pacer.AudioPacer`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from .logging import get_logger
from .tts import KEEPALIVE_TEXT

logger = get_logger("echobridge.keepalive")


class TtsKeepalive:
    def __init__(
        self,
        tts: Any,
        voice_id: str,
        interval_s: float,
        *,
        is_busy: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._tts = tts
        self._voice_id = voice_id
        self._interval_s = max(0.0, float(interval_s))
        self._is_busy = is_busy
        self._clock = clock
        self._last_attempt = 0.0
        self._failed = False

    @property
    def interval_s(self) -> float:
        return self._interval_s

    def _last_activity(self) -> float:
        tts_activity = float(getattr(self._tts, "last_synthesis_at", 0.0) or 0.0)
        return max(self._last_attempt, tts_activity)

    async def tick(self) -> bool:
        """Run one probe if it is due.  Returns True when TTS was called."""
        if self._interval_s <= 0:
            return False
        if self._is_busy():
            return False
        now = self._clock()
        if now - self._last_activity() < self._interval_s:
            return False
        self._last_attempt = now
        try:
            keepalive = getattr(self._tts, "keepalive", None)
            if callable(keepalive):
                await keepalive(self._voice_id)
            else:
                async for _ in self._tts.synthesize(
                    KEEPALIVE_TEXT, self._voice_id, 0
                ):
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A worker that is down stays down until systemd/keepalive fixes
            # it.  Log the transition once instead of every 10 minutes.
            if not self._failed:
                self._failed = True
                # Do not log the exception body: upstream may echo request
                # text.  The worker's own journal has the full stack.
                logger.warning(
                    "tts keepalive failed: %s", type(exc).__name__
                )
            return False
        if self._failed:
            logger.info("tts keepalive recovered")
            self._failed = False
        return True

    async def run(self) -> None:
        if self._interval_s <= 0:
            return
        # Check often enough that a short busy period does not skip a whole
        # interval; the tick itself enforces the configured gap.
        check_s = max(1.0, min(30.0, self._interval_s / 2.0))
        while True:
            await asyncio.sleep(check_s)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("tts keepalive loop failed")
