from __future__ import annotations

from echobridge.audio import (
    OUTPUT_FRAME_BYTES,
    iter_frames,
    pcm16_b64,
    rms_pcm16,
    silence_pcm16,
    tone_pcm16,
    trim_silence_pcm16,
)
from tests.helpers import loud_frame, silence_frame


def test_rms_silence_and_loud():
    assert rms_pcm16(silence_frame()) == 0
    assert rms_pcm16(loud_frame(9000)) > 700


def test_frames_are_padded_to_20ms():
    frames = list(iter_frames(b"\x01\x00" * 10, OUTPUT_FRAME_BYTES, pad=True))
    assert len(frames) == 1
    assert len(frames[0]) == 960


def test_tone_has_fade_and_audio():
    pcm = tone_pcm16("你好呀", char_ms=10)
    assert len(pcm) > 0
    assert rms_pcm16(pcm) > 0
    encoded = pcm16_b64(pcm)
    assert isinstance(encoded, str) and encoded


def test_empty_tone_is_empty():
    assert tone_pcm16("") == b""


def test_trim_silence_keeps_short_pads_and_inner_pause():
    voice = tone_pcm16("嗯", char_ms=200)
    pcm = (
        silence_pcm16(800)
        + voice
        + silence_pcm16(300)
        + voice
        + silence_pcm16(600)
    )
    trimmed = trim_silence_pcm16(pcm, lead_pad_ms=60, tail_pad_ms=100)
    # Two 200 ms tones + the 300 ms inner pause, plus at most the pads and
    # one 10 ms analysis window of slack on each side.
    assert 700 * 48 <= len(trimmed) <= (700 + 60 + 100 + 20) * 48
    assert trimmed[:2] == b"\x00\x00"  # lead pad is silence


def test_trim_silence_returns_silent_input_unchanged():
    pcm = silence_pcm16(500)
    assert trim_silence_pcm16(pcm) == pcm

