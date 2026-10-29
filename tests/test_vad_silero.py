"""SileroVAD start/pre-roll logic with a scripted probability model."""

from __future__ import annotations

import struct

from echobridge.vad import SileroVAD
from tests.helpers import make_settings

WINDOW_BYTES = 512 * 2


class ScriptedModel:
    """Returns one probability per 512-sample window, then 0.0."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self._i = 0

    def reset_states(self) -> None:
        return None

    def __call__(self, tensor, sample_rate: int) -> float:
        p = self._probs[self._i] if self._i < len(self._probs) else 0.0
        self._i += 1
        return p


def window(tag: int) -> bytes:
    """One Silero window whose samples all equal ``tag`` (to trace audio)."""
    return struct.pack("<h", tag) * 512


def silero_settings(**overrides):
    base = dict(
        vad_backend="silero",
        silero_threshold=0.5,
        silero_start_threshold=0.35,
        vad_start_ms=64,
        vad_start_gap_windows=1,
        vad_pre_roll_ms=800,
        vad_playback_pre_roll_ms=300,
        vad_endpoint_ms=320,
        vad_tail_pad_ms=0,
        vad_min_speech_ms=200,
        vad_short_carry_ms=1000,
    )
    base.update(overrides)
    return make_settings(**base)


def push_windows(vad: SileroVAD, tags: list[int]):
    events = []
    for tag in tags:
        events.extend(vad.push(window(tag)))
    return events


def tags_of(pcm: bytes) -> list[int]:
    return [
        struct.unpack_from("<h", pcm, offset)[0]
        for offset in range(0, len(pcm), WINDOW_BYTES)
    ]


def test_soft_onset_opens_with_start_threshold_and_one_dip():
    # 0.4 is below the 0.5 continuation threshold but above the idle start
    # threshold; the 0.2 dip must not reset the start run.
    probs = [0.0, 0.4, 0.2, 0.4] + [0.9] * 8 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    kinds = [e.kind for e in events]
    assert kinds[0] == "started"
    final = next(e for e in events if e.kind == "final")
    tags = tags_of(final.pcm)
    # Segment includes the silent window before the onset (pre-roll) and
    # every window exactly once.
    assert tags[:4] == [1, 2, 3, 4]
    assert len(tags) == len(set(tags))


def test_playback_uses_continuation_threshold_and_short_pre_roll():
    probs = [0.4] * 4 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    vad.set_playback(True)
    assert not [e for e in push_windows(vad, list(range(1, 17))) if e.kind == "started"]

    probs = [0.0] * 20 + [0.9] * 8 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    vad.set_playback(True)
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    final = next(e for e in events if e.kind == "final")
    tags = tags_of(final.pcm)
    # 300 ms pre-roll ~= 9 windows before the start window.
    assert tags[0] >= 21 - 10


def test_idle_pre_roll_covers_800ms():
    probs = [0.0] * 30 + [0.9] * 8 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    final = next(e for e in events if e.kind == "final")
    tags = tags_of(final.pcm)
    # Start fires on window 32 (two windows >= 64 ms); 800 ms ~= 25 windows.
    assert tags[0] <= 32 - 24


def test_long_utterance_trims_oldest_pre_roll():
    # 3.2 s of speech: keep the 800 ms VAD pre-roll for detection, but send
    # only the newest 300 ms to ASR (saves ~500 ms of encoder work).
    probs = [0.0] * 30 + [0.9] * 100 + [0.0] * 12
    vad = SileroVAD(
        silero_settings(vad_long_pre_roll_ms=300, vad_long_pre_roll_after_ms=2500),
        model=ScriptedModel(probs),
    )
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    final = next(e for e in events if e.kind == "final")
    assert final.speech_ms >= 2500
    tags = tags_of(final.pcm)
    # Start fires around window 32; only ~9 pre-roll windows (300 ms) remain.
    assert tags[0] >= 20
    assert tags[-1] >= 120


def test_pre_roll_does_not_duplicate_partial_frame():
    # Feed 20 ms frames (640 bytes) so the start window ends mid-frame.
    probs = [0.0] * 3 + [0.9] * 6 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    stream = b"".join(struct.pack("<h", i % 30000) for i in range(512 * len(probs)))
    events = []
    for offset in range(0, len(stream), 640):
        events.extend(vad.push(stream[offset : offset + 640]))
    final = next(e for e in events if e.kind == "final")
    samples = struct.unpack(f"<{len(final.pcm) // 2}h", final.pcm)
    assert all(b == a + 1 for a, b in zip(samples, samples[1:]))


def test_short_segment_is_carried_into_next_segment():
    # "停" (128 ms of speech, < min_speech) then a pause, then the rest.
    probs = [0.9] * 4 + [0.0] * 10 + [0.0] * 5 + [0.9] * 10 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 2
    assert finals[0].speech_ms < 200
    tags = tags_of(finals[1].pcm)
    assert tags[:4] == [1, 2, 3, 4]
    # Carried speech counts toward the next segment (the first start window
    # is not counted, as before).
    assert finals[1].speech_ms == finals[0].speech_ms + 9 * 32


def test_long_trim_keeps_short_carry():
    # Short "停" carried into a 3+ s segment; trimming the old pre-roll must
    # not delete the carried speech at the front.
    probs = [0.9] * 4 + [0.0] * 10 + [0.0] * 5 + [0.9] * 100 + [0.0] * 12
    vad = SileroVAD(
        silero_settings(vad_long_pre_roll_ms=300, vad_long_pre_roll_after_ms=2500),
        model=ScriptedModel(probs),
    )
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 2
    tags = tags_of(finals[1].pcm)
    assert tags[0] == 1
    assert tags[3] == 4


def test_short_segment_carry_expires():
    probs = [0.9] * 4 + [0.0] * 10 + [0.0] * 40 + [0.9] * 10 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    finals = [e for e in events if e.kind == "final"]
    assert 1 not in tags_of(finals[1].pcm)


def test_soft_voice_opened_idle_counts_as_speech():
    # Every window 0.4: opens at the idle start threshold and must keep
    # counting as speech instead of endpointing as silence at 0.5.
    probs = [0.0] * 3 + [0.4] * 20 + [0.0] * 12
    vad = SileroVAD(silero_settings(), model=ScriptedModel(probs))
    events = push_windows(vad, list(range(1, len(probs) + 1)))
    finals = [e for e in events if e.kind == "final"]
    assert len(finals) == 1
    assert finals[0].speech_ms >= 18 * 32
