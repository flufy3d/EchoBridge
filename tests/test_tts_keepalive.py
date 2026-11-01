from __future__ import annotations

import logging

from echobridge.keepalive import TtsKeepalive


class Clock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class CountingTTS:
    def __init__(self) -> None:
        self.calls = 0
        self.last_synthesis_at = 0.0

    async def keepalive(self, voice_id: str) -> None:
        self.calls += 1


class FailingTTS:
    def __init__(self) -> None:
        self.calls = 0
        self.last_synthesis_at = 0.0

    async def keepalive(self, voice_id: str) -> None:
        self.calls += 1
        raise RuntimeError("worker down")


async def test_keepalive_skips_active_response_and_respects_interval():
    tts = CountingTTS()
    clock = Clock()
    busy = False
    keepalive = TtsKeepalive(
        tts, "mingyue_v1", 600, is_busy=lambda: busy, clock=clock
    )

    # A response is in flight: no probe even though the interval elapsed.
    busy = True
    clock.now += 10_000
    assert await keepalive.tick() is False
    assert tts.calls == 0

    busy = False
    assert await keepalive.tick() is True
    assert tts.calls == 1

    # Same interval: an immediate tick must not synthesise again.
    assert await keepalive.tick() is False
    assert tts.calls == 1

    clock.now += 599
    assert await keepalive.tick() is False
    assert tts.calls == 1

    clock.now += 1
    assert await keepalive.tick() is True
    assert tts.calls == 2


async def test_keepalive_failure_logs_once_until_it_recovers(caplog):
    tts = FailingTTS()
    clock = Clock()
    keepalive = TtsKeepalive(
        tts, "mingyue_v1", 60, is_busy=lambda: False, clock=clock
    )

    with caplog.at_level(logging.WARNING, logger="echobridge.keepalive"):
        assert await keepalive.tick() is False
        clock.now += 60
        assert await keepalive.tick() is False
        clock.now += 60
        assert await keepalive.tick() is False

    assert tts.calls == 3
    failures = [
        record
        for record in caplog.records
        if "tts keepalive failed" in record.getMessage()
    ]
    assert len(failures) == 1
