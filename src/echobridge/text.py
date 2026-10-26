"""Text cleanup and sentence chunking for TTS."""

from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"[`*_#>\[\]]+")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def clean_for_tts(text: str) -> str:
    text = _URL_RE.sub("", text or "")
    text = _MARKDOWN_RE.sub("", text)
    # Drop emoji/symbols that a TTS model would read as noise, but keep
    # punctuation and currency/measurement symbols that may be meaningful.
    kept: list[str] = []
    for char in text:
        category = unicodedata.category(char)
        if category in {"So", "Sk"}:
            continue
        if char in "\ufe0f\u200d":
            continue
        kept.append(char)
    text = "".join(kept)
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def split_for_tts(text: str, first_max: int = 15, next_max: int = 40) -> list[str]:
    text = clean_for_tts(text)
    if not text:
        return []
    raw = [chunk.strip() for chunk in _SENTENCE_RE.findall(text) if chunk.strip()]
    if not raw:
        raw = [text]
    chunks: list[str] = []
    for sentence in raw:
        if len(sentence) <= (first_max if not chunks else next_max):
            chunks.append(sentence)
            continue
        limit = first_max if not chunks else next_max
        start = 0
        while start < len(sentence):
            end = min(len(sentence), start + limit)
            if end < len(sentence):
                # Prefer a comma/顿号 break near the limit.
                window = sentence[start:end]
                split_at = max(window.rfind("，"), window.rfind(","), window.rfind("、"))
                if split_at >= max(3, limit // 3):
                    end = start + split_at + 1
            piece = sentence[start:end].strip()
            if piece:
                chunks.append(piece)
            start = end
    return chunks
