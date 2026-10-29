"""Text cleanup and safe streaming chunking for TTS."""

from __future__ import annotations

import re
import unicodedata

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MARKDOWN_RE = re.compile(r"[`*_#>\[\]]+")
_SPACE_RE = re.compile(r"\s+")
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

# Sentence-ending punctuation.  A split here is always a natural pause.
_STRONG_PUNCT = frozenset("。！？!?；;\n")
# Clause punctuation.  Safe for a first low-latency chunk or a long clause.
_CLAUSE_PUNCT = frozenset("，,、：:～~…")
_ALL_PUNCT = _STRONG_PUNCT | _CLAUSE_PUNCT
# Characters after which a no-punctuation Chinese cut is usually comfortable.
_FUNCTION_WORDS = frozenset("的了吗呢吧呀啊和与在是有就都而但着过地得")
# Never emit a chunk shorter than this unless it is the final remainder.
_MIN_CHUNK_CHARS = 4


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


def _is_ascii_word_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in "_-'")


def _is_numeric_punct(text: str, index: int) -> bool:
    """True for comma/colon/tilde between digits (1,000 / 3:14 / 3~5)."""
    if index <= 0 or index + 1 >= len(text):
        return False
    return (
        text[index - 1].isdigit()
        and text[index + 1].isdigit()
        and text[index] in ",:：～~"
    )


def _safe_cut(text: str, index: int) -> bool:
    """True when splitting ``text[:index]`` / ``text[index:]`` is word-safe."""
    if index <= 0:
        return False
    if index >= len(text):
        # At the current end of stream, an ASCII letter/digit may be an
        # incomplete word ("Open" waiting for "AI"); wait for more deltas.
        return not _is_ascii_word_char(text[-1])
    left = text[index - 1]
    right = text[index]
    if _is_ascii_word_char(left) and _is_ascii_word_char(right):
        return False
    if (
        left.isdigit()
        and right in ".,．"
        and index + 1 < len(text)
        and text[index + 1].isdigit()
    ):
        return False
    if (
        left in ".,．"
        and index >= 2
        and text[index - 2].isdigit()
        and right.isdigit()
    ):
        return False
    if unicodedata.combining(right):
        return False
    return True


def _effective_count(text: str) -> int:
    return sum(
        1
        for char in text
        if not char.isspace() and char not in _ALL_PUNCT
    )


def _strip_leading_noise(text: str) -> str:
    text = text.lstrip()
    while text and text[0] in _ALL_PUNCT:
        text = text[1:].lstrip()
    return text


def _find_punct_index(
    text: str, punct: frozenset[str], min_effective: int
) -> int | None:
    effective = 0
    for index, char in enumerate(text):
        if char in punct:
            if _is_numeric_punct(text, index):
                continue
            if effective >= min_effective:
                return index
        elif not char.isspace() and char not in _ALL_PUNCT:
            effective += 1
    return None


def _consume_punct_run(text: str, index: int) -> int:
    """Consume repeated punctuation (……, 。」, etc.) after a split point."""
    index += 1
    while index < len(text) and text[index] in _ALL_PUNCT:
        index += 1
    return index


def _find_safe_hard_cut(text: str, cap: int) -> int | None:
    """Find a word-safe cut at/near ``cap`` without punctuation.

    Preference order: the last function-word boundary inside the cap, then the
    last whitespace, then any safe character boundary.  If the cap falls inside
    an ASCII word or number, scan forward for its natural end instead of
    breaking it.
    """
    limit = min(len(text), cap)
    # Function words first: "看到了" is a much better pause than "看到".
    for index in range(limit - 1, _MIN_CHUNK_CHARS - 1, -1):
        if text[index] in _FUNCTION_WORDS:
            cut = index + 1
            if cut >= _MIN_CHUNK_CHARS and _safe_cut(text, cut):
                return cut
    # Then whitespace (English words / mixed text).
    for index in range(limit - 1, _MIN_CHUNK_CHARS - 1, -1):
        if text[index].isspace():
            cut = index + 1
            if cut >= _MIN_CHUNK_CHARS:
                return cut
    # Chinese characters are safe at almost any character boundary; this is
    # the last resort before waiting for a natural delimiter.
    for index in range(limit, _MIN_CHUNK_CHARS - 1, -1):
        if _safe_cut(text, index):
            return index
    # The cap is inside a long ASCII token.  Wait until the token ends, then
    # cut after its delimiter rather than splitting "Open" + "AI".
    for index in range(cap, len(text)):
        if text[index].isspace() or text[index] in _ALL_PUNCT:
            cut = index + 1
            if _safe_cut(text, cut):
                return cut
    return None


def pop_tts_chunk(
    buffer: str,
    *,
    first_chunk: bool,
    first_soft_max: int = 15,
    first_hard_max: int = 22,
    next_clause_min: int = 24,
    next_hard_max: int = 36,
) -> tuple[str, str]:
    """Pop one safe TTS chunk, returning ``(chunk, remaining_buffer)``.

    The remaining buffer always keeps the incomplete tail after the split, so a
    comma never sends half a clause to TTS.  When no safe boundary exists yet
    (e.g. a long English token), it returns ``("", buffer)`` and the caller
    waits for more LLM deltas.
    """
    buffer = _strip_leading_noise(buffer)
    if not buffer:
        return "", ""

    if first_chunk:
        # Low-latency first clause: split at the earliest strong/clause
        # punctuation once at least four effective characters precede it.
        strong = _find_punct_index(buffer, _STRONG_PUNCT, _MIN_CHUNK_CHARS)
        clause = _find_punct_index(buffer, _CLAUSE_PUNCT, _MIN_CHUNK_CHARS)
        candidates = [index for index in (strong, clause) if index is not None]
        if candidates:
            end = _consume_punct_run(buffer, min(candidates))
            return buffer[:end].strip(), buffer[end:]

        effective = _effective_count(buffer)
        # Soft boundary preserves the old ~15-character first-audio latency
        # without ever ending in the middle of a word.
        if effective >= first_soft_max:
            cut = _find_safe_hard_cut(buffer, first_soft_max)
            if cut is not None:
                return buffer[:cut].strip(), buffer[cut:]
        if effective >= first_hard_max:
            cut = _find_safe_hard_cut(buffer, first_hard_max)
            if cut is not None:
                return buffer[:cut].strip(), buffer[cut:]
        return "", buffer

    # Subsequent chunks: complete sentences first.
    strong = _find_punct_index(buffer, _STRONG_PUNCT, 1)
    if strong is not None:
        end = _consume_punct_run(buffer, strong)
        return buffer[:end].strip(), buffer[end:]

    # A long clause with commas: split at the *last* clause punctuation and
    # keep everything after it for the next LLM delta.
    if _effective_count(buffer) >= next_clause_min:
        last = None
        for index, char in enumerate(buffer):
            if char in _CLAUSE_PUNCT and not _is_numeric_punct(buffer, index):
                last = index
        if last is not None and last + 1 >= _MIN_CHUNK_CHARS:
            end = _consume_punct_run(buffer, last)
            return buffer[:end].strip(), buffer[end:]

    # No punctuation: only cut at the hard cap, at a word/function boundary.
    if _effective_count(buffer) >= next_hard_max:
        cut = _find_safe_hard_cut(buffer, next_hard_max)
        if cut is not None:
            return buffer[:cut].strip(), buffer[cut:]
    return "", buffer


def _has_speakable_text(text: str) -> bool:
    return any(
        not char.isspace() and char not in _ALL_PUNCT for char in text
    )


def split_for_tts(
    text: str,
    first_max: int = 22,
    next_max: int = 36,
    clause_min: int = 24,
) -> list[str]:
    """Split a complete string with the same safe boundaries as streaming.

    Used for the final flush after the LLM stream ends; it never cuts inside an
    English word or a decimal number.
    """
    text = clean_for_tts(text)
    if not text:
        return []
    chunks: list[str] = []
    buffer = text
    first = True
    while buffer.strip():
        chunk, remaining = pop_tts_chunk(
            buffer,
            first_chunk=first,
            first_hard_max=first_max,
            next_hard_max=next_max,
            next_clause_min=clause_min,
        )
        if not chunk:
            break
        if _has_speakable_text(chunk):
            chunks.append(chunk)
        buffer = remaining
        first = False
    tail = buffer.strip()
    if tail and _has_speakable_text(tail):
        chunks.append(tail)
    return chunks
