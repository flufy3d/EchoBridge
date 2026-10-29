from __future__ import annotations

import base64

from echobridge import protocol
from echobridge.audio import OUTPUT_FRAME_BYTES
from echobridge.pacer import AudioPacer


async def _collect(pcm_chunks: list[bytes]) -> tuple[list[dict], bytes]:
    events: list[dict] = []

    async def send(event: dict, ctx: protocol.ResponseContext) -> bool:
        events.append(event)
        return True

    ctx = protocol.ResponseContext(response_id="resp_1", generation=1)
    pacer = AudioPacer(send, ctx, interval=0.0)
    pacer.start()
    for chunk in pcm_chunks:
        await pacer.feed(chunk)
    await pacer.finish()
    audio = b"".join(
        base64.b64decode(event["delta"])
        for event in events
        if event["type"] == protocol.AUDIO_DELTA
    )
    return events, audio


async def test_pacer_carries_partial_frames_between_chunks():
    # The real vLLM client reads 65529 / 30471 byte blocks.  Padding each block
    # inserted ~15-20 ms of silence every two seconds; the carried tail must
    # reconstruct the input exactly.
    first = bytes((index * 7) % 256 for index in range(65529))
    second = bytes((index * 11) % 256 for index in range(30471))
    events, audio = await _collect([first, second])

    assert len(first) + len(second) == 96_000
    assert audio == first + second
    deltas = [event for event in events if event["type"] == protocol.AUDIO_DELTA]
    assert len(deltas) == 100
    assert all(len(base64.b64decode(item["delta"])) == OUTPUT_FRAME_BYTES for item in deltas)
    assert events[0]["type"] == protocol.AUDIO_STARTED
    assert events[-1]["type"] == protocol.AUDIO_DONE


async def test_pacer_pads_only_the_final_frame():
    pcm = b"\x01\x00" * 1_000  # 2000 bytes = two frames + 80 bytes
    events, audio = await _collect([pcm])
    assert audio[: len(pcm)] == pcm
    assert audio[len(pcm) :] == b"\x00" * (OUTPUT_FRAME_BYTES - len(pcm) % OUTPUT_FRAME_BYTES)
    assert len(audio) == 2_880
