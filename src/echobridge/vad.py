"""Phase 1 energy VAD.

This is deliberately not Silero.  It exists to exercise the protocol and the
real device audio path; Phase 2 replaces it with Silero VAD plus pre-roll and
proper endpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .audio import rms_pcm16
from .config import Settings


@dataclass(slots=True)
class VADEvent:
    kind: Literal["started", "delta", "final"]
    item_id: str
    text: str = ""
    speech_ms: int = 0


class EnergyVAD:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._consecutive = 0
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._delta_sent = False
        self._item_seq = 0
        self._item_id = ""
        self._text = ""
        self._frame_ms = 20

    @property
    def active(self) -> bool:
        return self._active

    def reset(self) -> None:
        self._consecutive = 0
        self._active = False
        self._speech_ms = 0
        self._silence_ms = 0
        self._delta_sent = False
        self._item_id = ""
        self._text = ""

    def push(self, pcm: bytes, transcript_hint: str) -> list[VADEvent]:
        events: list[VADEvent] = []
        loud = rms_pcm16(pcm) >= self._settings.vad_threshold

        if not self._active:
            if loud:
                self._consecutive += 1
                if self._consecutive < self._settings.vad_start_frames:
                    return events
                self._active = True
                self._item_seq += 1
                self._item_id = f"item_{self._item_seq}"
                self._speech_ms = 0
                self._silence_ms = 0
                self._delta_sent = False
                self._text = transcript_hint
                events.append(VADEvent("started", self._item_id))
            else:
                self._consecutive = 0
                return events

        # Active segment: the frame that triggered onset is also speech.
        if loud:
            self._speech_ms += self._frame_ms
            self._silence_ms = 0
            if not self._delta_sent and self._speech_ms >= 400 and self._text:
                self._delta_sent = True
                prefix_len = max(1, len(self._text) // 2)
                events.append(
                    VADEvent("delta", self._item_id, self._text[:prefix_len], self._speech_ms)
                )
        else:
            self._silence_ms += self._frame_ms

        if self._silence_ms >= self._settings.vad_endpoint_ms:
            events.append(self._final())
        elif self._speech_ms >= self._settings.vad_max_speech_ms:
            events.append(self._final())
        return events

    def _final(self) -> VADEvent:
        item_id = self._item_id
        speech_ms = self._speech_ms
        text = self._text if speech_ms >= self._settings.vad_min_speech_ms else ""
        self.reset()
        return VADEvent("final", item_id, text, speech_ms)
