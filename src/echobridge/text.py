"""Text cleanup and sentence chunking for TTS."""

from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"[`*_#>\[\]]+")
_SPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
_STOP_RE = re.compile(
    r"^(?:"
    r"停(?:一下|下|一停|停|下来)?"
    r"|别(?:说|讲)了"
    r"|不要(?:说|讲)了"
    r"|不用(?:说|讲)了"
    r"|不用了"
    r"|算了"
    r"|等等"
    r"|等一下"
    r"|打断一下"
    r"|不听了"
    r"|stop|wait|hold\s*on|shut\s*up"
    r")[。！？!?，,、；;：:\s吧啊呀~～]*$",
    re.IGNORECASE,
)


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


def is_stop_command(text: str) -> bool:
    """True for short utterances that mean "stop talking", not a new request."""
    cleaned = clean_for_tts(text or "").strip()
    if not cleaned or len(cleaned) > 16:
        return False
    # Half-duplex barge-in can clip the first syllable ("停一" lost), leaving
    # fragments such as "下，不用讲了。".  Strip leading fillers/punctuation and
    # also accept an unambiguous stop phrase anywhere in a short utterance.
    normalized = re.sub(r"^[下一好嗯呃啊呀那个，,、\s]+", "", cleaned)
    if _STOP_RE.match(normalized):
        return True
    tail = re.sub(r"[。！？!?，,、；;：:\s吧啊呀~～]+$", "", cleaned)
    for phrase in (
        "不用讲了",
        "不用说了",
        "别讲了",
        "别说了",
        "不要讲了",
        "不要说了",
        "停一下",
        "停下来",
        "停下",
        "不听了",
    ):
        if tail.endswith(phrase):
            return True
    return False


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
