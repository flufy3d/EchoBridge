from __future__ import annotations

from echobridge.vad import EnergyVAD
from tests.helpers import loud_frame, make_settings, silence_frame


def test_energy_vad_pre_roll_and_tail_trim():
    settings = make_settings(
        vad_endpoint_ms=100,
        vad_min_speech_ms=40,
        vad_pre_roll_ms=60,
        vad_tail_pad_ms=20,
        vad_start_frames=2,
    )
    vad = EnergyVAD(settings)
    events = []
    for _ in range(5):
        events.extend(vad.push(loud_frame()))
    for _ in range(5):
        events.extend(vad.push(silence_frame()))
    kinds = [event.kind for event in events]
    assert kinds[0] == "started"
    assert kinds[-1] == "final"
    final = events[-1]
    assert final.speech_ms == 80
    # 2 frames of pre-roll + 80 ms speech + 20 ms tail pad = ~120 ms.
    assert 0.10 < len(final.pcm) / 2 / 16000 < 0.14


def test_energy_vad_short_speech_is_reported_empty():
    settings = make_settings(
        vad_endpoint_ms=100,
        vad_min_speech_ms=200,
        vad_start_frames=2,
    )
    vad = EnergyVAD(settings)
    events = []
    for _ in range(3):
        events.extend(vad.push(loud_frame()))
    for _ in range(5):
        events.extend(vad.push(silence_frame()))
    assert events[-1].kind == "final"
    assert events[-1].speech_ms < 200
    assert len(events[-1].pcm) > 0
