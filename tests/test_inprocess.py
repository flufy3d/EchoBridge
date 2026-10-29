from __future__ import annotations

import asyncio

import pytest

from echobridge.app import _worker_health
from echobridge.asr import ASRResult, InProcessQwen3ASR, make_asr
from echobridge.config import Settings
from echobridge.gpu import GPUScheduler
from echobridge.session import Session
from echobridge.tts import make_tts
from echobridge.vad import VADEvent
from tests.fakes import FakeWebSocket
from tests.helpers import make_settings, make_voice_registry


def test_make_asr_selects_inprocess_without_loading():
    settings = make_settings(asr_backend="inprocess")
    backend = make_asr(settings)
    assert isinstance(backend, InProcessQwen3ASR)
    assert backend.backend == "inprocess"
    assert backend.loaded is False
    assert backend.health()["ok"] is False


def test_make_tts_selects_inprocess_without_loading(tmp_path):
    registry = make_voice_registry(tmp_path)
    settings = make_settings(
        tts_backend="inprocess",
        tts_voice_dir=str(tmp_path),
    )
    backend = make_tts(settings, registry)
    assert backend.backend == "inprocess"
    assert backend.health()["ok"] is False
    assert backend.cached("你好", "unknown_voice") is None


async def test_gpu_scheduler_prioritises_asr_over_tts():
    scheduler = GPUScheduler()
    order: list[str] = []

    async def worker(name: str, priority: int, hold: float) -> None:
        async with scheduler.lease(name, priority=priority):
            order.append(name)
            await asyncio.sleep(hold)

    async with scheduler.lease("main", priority=100):
        tts = asyncio.create_task(worker("tts", 10, 0.01))
        await asyncio.sleep(0.02)
        asr = asyncio.create_task(worker("asr", 30, 0.01))
        await asyncio.sleep(0.02)
        assert order == []
    await asyncio.gather(tts, asr)
    assert order == ["asr", "tts"]


async def test_readyz_worker_health_reports_inprocess_without_http():
    class Backend:
        def health(self):
            return {"ok": True, "backend": "inprocess"}

    settings = make_settings(
        asr_backend="inprocess", tts_backend="inprocess"
    )
    workers = await _worker_health(
        settings, asr=Backend(), tts=Backend()
    )
    assert workers["asr"]["ok"] is True
    assert workers["tts"]["ok"] is True
    assert workers["asr"]["backend"] == "inprocess"


class CountingASR:
    def __init__(self) -> None:
        self.calls: list[bytes] = []

    async def transcribe(self, pcm, sample_rate=16000, context=None, **kwargs):
        self.calls.append(pcm)
        await asyncio.sleep(0.02)
        return ASRResult("你好", "zh", 20.0)

    async def warmup(self) -> None:
        return None

    async def close(self) -> None:
        return None


async def test_speculative_result_is_reused_once(tmp_path):
    registry = make_voice_registry(tmp_path)
    asr = CountingASR()
    session = Session(
        FakeWebSocket(),
        make_settings(
            asr_backend="mock",
            tts_backend="mock",
            vad_backend="energy",
            tts_voice_dir=str(tmp_path),
        ),
        asr=asr,
        tts=make_tts(
            make_settings(tts_backend="mock", tts_voice_dir=str(tmp_path)),
            registry,
        ),
        voice_registry=registry,
    )
    pcm = b"\x01\x00" * 1600
    session._start_speculative_asr(VADEvent("early", "item_1", pcm=pcm))
    result = await session._take_speculative_asr(
        VADEvent("final", "item_1", pcm=pcm + b"\x00\x00" * 1600)
    )
    assert result is not None and result.text == "你好"
    assert len(asr.calls) == 1
    assert session._spec_task is None

    # A second final with no in-flight speculative task must not reuse stale
    # state; the session will run its own transcription.
    assert await session._take_speculative_asr(
        VADEvent("final", "item_1", pcm=pcm)
    ) is None
