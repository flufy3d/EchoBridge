from __future__ import annotations

import asyncio

from echobridge.llm import LLMDelta
from echobridge.speculative import SpeculativeLLM


async def test_speculative_replays_every_buffered_delta():
    async def source():
        for text in ("你", "好", "呀"):
            await asyncio.sleep(0)
            yield LLMDelta(text=text)
        yield LLMDelta(done=True)

    spec = SpeculativeLLM(source, name="test")
    spec.start()
    # Let the producer run ahead so the consumer is replaying a buffer, not
    # simply following the live stream.
    await asyncio.sleep(0.05)

    deltas = [delta async for delta in spec.deltas()]
    assert [delta.text for delta in deltas if delta.text] == ["你", "好", "呀"]
    assert deltas[-1].done
    assert spec.produced == 4
    assert spec.error is None


async def test_closing_the_consumer_cancels_the_producer():
    cancelled = asyncio.Event()

    async def source():
        try:
            while True:
                await asyncio.sleep(0.01)
                yield LLMDelta(text="x")
        except asyncio.CancelledError:
            cancelled.set()
            raise

    spec = SpeculativeLLM(source, name="cancel")
    stream = spec.deltas()
    await anext(stream)
    await stream.aclose()
    await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    assert spec.task is not None and spec.task.cancelled()
