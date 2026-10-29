"""Voice activity detection.

Two implementations share one streaming interface:
  * EnergyVAD  - Phase 1 fallback/test double, 20 ms frames.
  * SileroVAD  - Phase 2 production VAD, 512-sample windows on CPU.

Both return the complete speech segment (including the pre-roll) so the
ASR worker can transcribe it without keeping a second audio buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal, Protocol

from .audio import rms_pcm16
from .config import Settings


@dataclass(slots=True)
class VADEvent:
    kind: Literal["started", "delta", "early", "resume", "final"]
    item_id: str
    text: str = ""
    speech_ms: int = 0
    pcm: bytes = b""


class VADBackend(Protocol):
    @property
    def active(self) -> bool: ...

    def reset(self) -> None: ...

    def set_playback(self, playing: bool) -> None: ...

    def push(self, pcm: bytes) -> list[VADEvent]: ...


class EnergyVAD:
    """Deterministic energy gate used by tests and as a fallback."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._frame_ms = 20
        self._pre_roll: deque[bytes] = deque(
            maxlen=max(1, settings.vad_pre_roll_ms // self._frame_ms)
        )
        self._segment = bytearray()
        self._consecutive = 0
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._item_seq = 0
        self._item_id = ""

    @property
    def active(self) -> bool:
        return self._active

    def set_playback(self, playing: bool) -> None:
        # The energy gate has no start/pre-roll split; kept for the interface.
        return None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._segment.clear()
        self._consecutive = 0
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._item_id = ""

    def push(self, pcm: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        if not pcm:
            return events
        self._pre_roll.append(bytes(pcm))
        loud = rms_pcm16(pcm) >= self._settings.vad_threshold
        started_now = False

        if not self._active:
            if not loud:
                self._consecutive = 0
                return events
            self._consecutive += 1
            if self._consecutive < self._settings.vad_start_frames:
                return events
            self._active = True
            self._item_seq += 1
            self._item_id = f"item_{self._item_seq}"
            self._segment = bytearray(b"".join(self._pre_roll))
            self._speech_ms = 0
            self._silence_ms = 0
            started_now = True
            events.append(VADEvent("started", self._item_id))

        if not started_now:
            self._segment.extend(pcm)

        if loud:
            self._speech_ms += self._frame_ms
            self._silence_ms = 0
        else:
            self._silence_ms += self._frame_ms

        if self._silence_ms >= self._settings.vad_endpoint_ms:
            events.append(self._final())
        elif self._speech_ms >= self._settings.vad_max_speech_ms:
            events.append(self._final())
        return events

    def _trimmed_pcm(self, silence_ms: int) -> bytes:
        trim_ms = max(0, silence_ms - self._settings.vad_tail_pad_ms)
        pcm = bytes(self._segment)
        if trim_ms:
            cut = min(len(pcm), trim_ms * 32)
            pcm = pcm[: len(pcm) - cut]
        return pcm

    def _final(self) -> VADEvent:
        event = VADEvent(
            "final",
            self._item_id,
            speech_ms=self._speech_ms,
            pcm=self._trimmed_pcm(self._silence_ms),
        )
        self.reset()
        return event


class SileroVAD:
    """Silero VAD (ONNX, CPU) with explicit pre-roll and endpointing.

    The model is loaded once and can be shared by the single-device session
    manager.  State is reset between sessions/turns.
    """

    _WINDOW_SAMPLES = 512
    _WINDOW_BYTES = _WINDOW_SAMPLES * 2
    _WINDOW_MS = 32

    def __init__(self, settings: Settings, model=None) -> None:
        self._settings = settings
        self._threshold = float(settings.silero_threshold)
        # Opening a segment while idle uses a lower threshold: a soft first
        # clause ("没什么，", "停一下，") on VOICE_COMMUNICATION audio often
        # scores 0.3-0.5 and was never detected.  During playback the AEC
        # residual must not open segments, so the normal threshold applies.
        self._start_threshold = min(
            self._threshold, float(settings.silero_start_threshold)
        )
        self._frame_ms = 20
        self._pre_roll: deque[bytes] = deque(
            maxlen=max(
                1,
                max(settings.vad_pre_roll_ms, settings.vad_playback_pre_roll_ms)
                // self._frame_ms,
            )
        )
        self._playback = False
        self._window = bytearray()
        self._segment = bytearray()
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._start_windows = 0
        self._start_gaps = 0
        self._item_seq = 0
        self._item_id = ""
        self._early_sent = False
        # Too-short segment kept for the next one (see vad_short_carry_ms).
        self._carry = b""
        self._carry_speech_ms = 0
        self._carry_age_ms = 0
        if model is not None:
            self._model = model
        else:
            from silero_vad import load_silero_vad
            from silero_vad.utils_vad import OnnxWrapper

            if settings.silero_model:
                self._model = OnnxWrapper(str(settings.silero_model), force_onnx_cpu=True)
            else:
                self._model = load_silero_vad(onnx=True)
        self._reset_model()
        # Lazy imports keep the mock/energy path free of torch imports.
        import numpy as np
        import torch

        self._np = np
        self._torch = torch

    def _reset_model(self) -> None:
        try:
            self._model.reset_states()
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self._active

    def set_playback(self, playing: bool) -> None:
        self._playback = playing

    def reset(self) -> None:
        self._pre_roll.clear()
        self._window.clear()
        self._segment.clear()
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._start_windows = 0
        self._start_gaps = 0
        self._item_id = ""
        self._early_sent = False
        self._carry = b""
        self._carry_speech_ms = 0
        self._carry_age_ms = 0
        self._reset_model()

    def _pre_roll_pcm(self) -> bytes:
        """Audio before the start window, excluding bytes still in _window.

        The newest pre-roll frame may be only partly consumed into Silero
        windows; its unconsumed tail is appended again by later windows, so
        drop it here to avoid duplicating 0-20 ms at the segment start.
        """
        pcm = b"".join(self._pre_roll)
        if self._window:
            pcm = pcm[: max(0, len(pcm) - len(self._window))]
        pre_roll_ms = (
            self._settings.vad_playback_pre_roll_ms
            if self._playback
            else self._settings.vad_pre_roll_ms
        )
        keep = pre_roll_ms * 32
        return pcm[-keep:] if len(pcm) > keep else pcm

    def _probability(self, window: bytes) -> float:
        samples = self._np.frombuffer(window, dtype=self._np.int16).astype(self._np.float32)
        samples /= 32768.0
        tensor = self._torch.from_numpy(samples)
        return float(self._model(tensor, 16000))

    def push(self, pcm: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        if not pcm:
            return events
        self._pre_roll.append(bytes(pcm))
        self._window.extend(pcm)
        if self._carry and not self._active:
            self._carry_age_ms += len(pcm) // 32
            if self._carry_age_ms > self._settings.vad_short_carry_ms:
                self._carry = b""
                self._carry_speech_ms = 0

        while len(self._window) >= self._WINDOW_BYTES:
            window = bytes(self._window[: self._WINDOW_BYTES])
            del self._window[: self._WINDOW_BYTES]
            probability = self._probability(window)
            started_now = False

            if not self._active:
                start_threshold = (
                    self._threshold if self._playback else self._start_threshold
                )
                if probability >= start_threshold:
                    self._start_windows += 1
                    self._start_gaps = 0
                    if self._start_windows * self._WINDOW_MS >= self._settings.vad_start_ms:
                        self._active = True
                        self._item_seq += 1
                        self._item_id = f"item_{self._item_seq}"
                        self._segment = bytearray(self._carry + self._pre_roll_pcm())
                        self._speech_ms = self._carry_speech_ms
                        self._silence_ms = 0
                        self._start_gaps = 0
                        self._carry = b""
                        self._carry_speech_ms = 0
                        self._carry_age_ms = 0
                        started_now = True
                        events.append(VADEvent("started", self._item_id))
                elif (
                    self._start_windows
                    and self._start_gaps < self._settings.vad_start_gap_windows
                ):
                    # One dip inside a soft onset must not restart the count.
                    self._start_gaps += 1
                else:
                    self._start_windows = 0
                    self._start_gaps = 0

            if not self._active:
                continue
            if not started_now:
                self._segment.extend(window)

            if probability >= self._threshold:
                if self._early_sent:
                    self._early_sent = False
                    events.append(VADEvent("resume", self._item_id, speech_ms=self._speech_ms))
                self._speech_ms += self._WINDOW_MS
                self._silence_ms = 0
            else:
                self._silence_ms += self._WINDOW_MS

            # Start ASR slightly before the official endpoint, while the user is
            # already silent.  If speech resumes, the session discards it.
            early_at = max(
                0,
                self._settings.vad_endpoint_ms - self._settings.vad_speculative_lead_ms,
            )
            if (
                not self._early_sent
                and self._speech_ms >= self._settings.vad_min_speech_ms
                and self._silence_ms >= early_at
            ):
                self._early_sent = True
                events.append(
                    VADEvent(
                        "early",
                        self._item_id,
                        speech_ms=self._speech_ms,
                        pcm=self._trimmed_pcm(self._silence_ms),
                    )
                )

            if self._silence_ms >= self._settings.vad_endpoint_ms:
                events.append(self._final())
                break
            if self._speech_ms >= self._settings.vad_max_speech_ms:
                events.append(self._final())
                break
        return events

    def _trimmed_pcm(self, silence_ms: int) -> bytes:
        trim_ms = max(0, silence_ms - self._settings.vad_tail_pad_ms)
        pcm = bytes(self._segment)
        if trim_ms:
            cut = min(len(pcm), trim_ms * 32)
            pcm = pcm[: len(pcm) - cut]
        return pcm

    def _final(self) -> VADEvent:
        event = VADEvent(
            "final",
            self._item_id,
            speech_ms=self._speech_ms,
            pcm=self._trimmed_pcm(self._silence_ms),
        )
        self.reset()
        if (
            not self._playback
            and event.pcm
            and event.speech_ms < self._settings.vad_min_speech_ms
        ):
            # The session drops this as empty speech; keep it so a lone "停"
            # followed by a pause still reaches ASR with the next segment.
            self._carry = event.pcm
            self._carry_speech_ms = event.speech_ms
        return event


def make_vad(settings: Settings) -> VADBackend:
    if settings.vad_backend == "silero":
        return SileroVAD(settings)
    return EnergyVAD(settings)
