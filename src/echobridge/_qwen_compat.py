"""Compatibility shims for optional upstream imports.

``qwen-tts`` imports its 25 Hz tokenizer at package import time.  That branch
imports ``torchaudio.compliance.kaldi`` even when only the 12 Hz tokenizer is
used.  The gateway venv intentionally does not install torchaudio (its binary
wheel is tied to a specific torch build and would downgrade the CUDA stack), so
we provide an import-only stub for the unused branch.  Any real call fails
loudly instead of silently producing wrong audio.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types


def _module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__package__ = name.rpartition(".")[0]
    return module


def ensure_torchaudio_stub() -> None:
    try:
        import torchaudio  # noqa: F401
    except Exception:
        pass
    else:
        return

    torchaudio = _module("torchaudio")
    torchaudio.__version__ = "0.0.0+echobridge-stub"
    compliance = _module("torchaudio.compliance")
    kaldi = _module("torchaudio.compliance.kaldi")

    def _unavailable(*_: object, **__: object) -> None:
        raise RuntimeError(
            "torchaudio.compliance.kaldi is unavailable in this environment; "
            "the 25 Hz Qwen3-TTS tokenizer is not supported"
        )

    kaldi.fbank = _unavailable
    kaldi.spectrogram = _unavailable
    torchaudio.compliance = compliance
    compliance.kaldi = kaldi
    sys.modules.setdefault("torchaudio", torchaudio)
    sys.modules.setdefault("torchaudio.compliance", compliance)
    sys.modules.setdefault("torchaudio.compliance.kaldi", kaldi)
