"""PCM helpers and a deterministic mock TTS source.

No audio is written to disk by the gateway.  The Phase 1 mock uses short
pentatonic tones so the real device can verify the full transport/playback path
without shipping reference audio or model weights.
"""

from __future__ import annotations

import array
import base64
import math
import sys
import wave
from pathlib import Path


MIC_FRAME_BYTES = 640  # 20 ms @ 16 kHz mono PCM16
OUTPUT_FRAME_BYTES = 960  # 20 ms @ 24 kHz mono PCM16


def rms_pcm16(pcm: bytes) -> int:
    if not pcm or len(pcm) < 2:
        return 0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0
    total = 0
    for sample in samples:
        total += int(sample) * int(sample)
    return min(32767, int(math.sqrt(total / len(samples))))


def pcm16_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def iter_frames(pcm: bytes, frame_bytes: int = OUTPUT_FRAME_BYTES, pad: bool = True):
    for offset in range(0, len(pcm), frame_bytes):
        frame = pcm[offset : offset + frame_bytes]
        if pad and len(frame) < frame_bytes:
            frame = frame + b"\x00" * (frame_bytes - len(frame))
        if frame:
            yield frame


def pcm16_duration_ms(pcm: bytes, sample_rate: int = 24_000) -> float:
    return (len(pcm) / 2) / sample_rate * 1000.0


def silence_pcm16(milliseconds: int, sample_rate: int = 24_000) -> bytes:
    samples = max(1, int(sample_rate * milliseconds / 1000))
    return b"\x00\x00" * samples


def trim_silence_pcm16(
    pcm: bytes,
    sample_rate: int = 24_000,
    threshold: int = 300,
    window_ms: int = 10,
    lead_pad_ms: int = 60,
    tail_pad_ms: int = 100,
) -> bytes:
    """Trim leading/trailing silence, keeping a short pad on each side.

    Qwen3-TTS often renders a single-syllable filler ("嗯～") as 0.3 s of
    voice wrapped in up to 1 s of silence; the leading part is pure latency.
    Inner pauses are kept.  Returns ``pcm`` unchanged if nothing is voiced.
    """
    pcm = pcm[: len(pcm) - (len(pcm) % 2)]
    window = max(2, int(sample_rate * window_ms / 1000) * 2)
    voiced = [
        offset
        for offset in range(0, len(pcm), window)
        if rms_pcm16(pcm[offset : offset + window]) >= threshold
    ]
    if not voiced:
        return pcm
    lead_pad = int(sample_rate * lead_pad_ms / 1000) * 2
    tail_pad = int(sample_rate * tail_pad_ms / 1000) * 2
    start = max(0, voiced[0] - lead_pad)
    end = min(len(pcm), voiced[-1] + window + tail_pad)
    return pcm[start:end]


def linear_resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate or not pcm:
        return pcm
    source = array.array("h")
    source.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if sys.byteorder != "little":
        source.byteswap()
    if not source:
        return b""
    target_count = max(1, int(round(len(source) * target_rate / source_rate)))
    output = array.array("h", [0]) * target_count
    last = len(source) - 1
    ratio = source_rate / target_rate
    for i in range(target_count):
        position = i * ratio
        left = int(position)
        frac = position - left
        right = min(left + 1, last)
        value = int(source[left] * (1.0 - frac) + source[right] * frac)
        output[i] = max(-32768, min(32767, value))
    if sys.byteorder != "little":
        output.byteswap()
    return output.tobytes()


def load_wav_mono_24k(path: str | Path) -> bytes:
    """Load a PCM16 mono WAV and resample to 24 kHz.

    Phase 1 only needs test audio.  If the file is stereo we take the first
    channel; this is deliberately forgiving because it is not production TTS.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError("mock TTS WAV must be PCM16")
    if channels > 1:
        samples = array.array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        mono = samples[::channels]
        frames = mono.tobytes()
        if sys.byteorder != "little":
            # The array was byteswapped for host order; convert back.
            mono.byteswap()
            frames = mono.tobytes()
    return linear_resample_pcm16(frames, rate, 24_000)


_PENTATONIC = (523.25, 587.33, 659.25, 783.99, 880.00)


def tone_pcm16(
    text: str,
    sample_rate: int = 24_000,
    char_ms: int = 55,
    gap_ms: int = 8,
    amplitude: float = 0.18,
    note_offset: int = 0,
) -> bytes:
    """Generate a short, pleasant, speech-length tone sequence.

    This is a transport test signal, not speech.  Each non-space character gets
    one plucked note; punctuation gets a lower note.  The result has clean fades
    so the device's speaker path can be checked for clicks/pops.  ``note_offset``
    shifts the pentatonic sequence so mock voices are audibly distinct.
    """
    if not text:
        return b""
    samples = array.array("h")
    sample_count = max(1, int(sample_rate * char_ms / 1000))
    gap_count = max(0, int(sample_rate * gap_ms / 1000))
    fade = max(1, int(sample_rate * 0.006))

    for index, char in enumerate(text):
        if char.isspace():
            base = 0.0
        else:
            base = _PENTATONIC[
                (ord(char) + index + note_offset) % len(_PENTATONIC)
            ]
            if char in "。！？!?；;，,、":
                base *= 0.75
        if base <= 0.0:
            samples.extend([0] * max(1, sample_count // 2))
            continue
        angular = 2.0 * math.pi * base / sample_rate
        for i in range(sample_count):
            envelope = min(1.0, min(i / fade, (sample_count - 1 - i) / fade))
            value = math.sin(angular * i) * 0.75 + math.sin(angular * 2.0 * i) * 0.25
            samples.append(int(value * amplitude * 32767.0 * envelope))
        samples.extend([0] * gap_count)

    if not samples:
        return b""
    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()
