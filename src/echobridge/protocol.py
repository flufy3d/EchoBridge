"""Doubao Realtime protocol subset used by the Mingyue Live Android app.

The Android client ignores unknown fields, but it *does* depend on a handful of
exact field names and state transitions.  Keep this module as the single place
where wire-format decisions live.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Upstream event names
# ---------------------------------------------------------------------------

SESSION_CREATE = "session.create"
AUDIO_APPEND = "input_audio_buffer.append"
SPEECH_APPEND = "speech_text_buffer.append"
SPEECH_COMMIT = "speech_text_buffer.commit"
ITEM_CREATE = "conversation.item.create"
RESPONSE_CANCEL = "response.cancel"
SESSION_CLOSE = "session.close"


# ---------------------------------------------------------------------------
# Downstream event names
# ---------------------------------------------------------------------------

SESSION_CREATED = "session.created"
SESSION_UPDATED = "session.updated"
SESSION_CLOSED = "session.closed"
ASR_STARTED = "conversation.item.input_audio_transcription.started"
ASR_DELTA = "conversation.item.input_audio_transcription.delta"
ASR_COMPLETED = "conversation.item.input_audio_transcription.completed"
ASR_FAILED = "conversation.item.input_audio_transcription.failed"
TEXT_DELTA = "response.output_text.delta"
TEXT_DONE = "response.output_text.done"
AUDIO_STARTED = "response.output_audio.started"
AUDIO_DELTA = "response.output_audio.delta"
AUDIO_DONE = "response.output_audio.done"
TOOL_CALL_DONE = "response.function_call_arguments.done"
RESPONSE_DONE = "response.done"
RESPONSE_CANCELED = "response.canceled"
BACKCHANNEL = "response.backchannel"
ERROR = "error"


@dataclass(slots=True)
class ResponseContext:
    """Identity of one model/TTS response.

    The Android app does not currently read these fields, but carrying them on
    every response-scoped event makes cancel/discard logic explicit and testable.
    """

    response_id: str
    generation: int
    text: str = ""
    audio_started: bool = False
    text_done: bool = False
    done: bool = False


@dataclass(slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: str


@dataclass(slots=True)
class ToolResult:
    call_id: str
    text: str


def dumps(event: dict[str, Any]) -> str:
    # Compact JSON keeps 20 ms audio frames small and avoids ASCII escapes for
    # Chinese text.  The app uses org.json and accepts both forms.
    return json.dumps(event, ensure_ascii=False, separators=(",", ":"))


def parse_event(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("event must be a JSON object")
    return value


# ---------------------------------------------------------------------------
# Event builders
# ---------------------------------------------------------------------------


def session_created(session_id: str, model: str, backend: str = "echobridge-mock") -> dict[str, Any]:
    return {
        "type": SESSION_CREATED,
        "session": {
            "id": session_id,
            "model": model,
            "backend": backend,
        },
    }


def session_updated(session_id: str, model: str, backend: str = "echobridge-mock") -> dict[str, Any]:
    return {
        "type": SESSION_UPDATED,
        "session": {"id": session_id, "model": model, "backend": backend},
    }


def session_closed(reason: str = "client") -> dict[str, Any]:
    return {"type": SESSION_CLOSED, "reason": reason}


def asr_started(item_id: str) -> dict[str, Any]:
    return {"type": ASR_STARTED, "item_id": item_id}


def asr_delta(item_id: str, delta: str) -> dict[str, Any]:
    return {"type": ASR_DELTA, "item_id": item_id, "delta": delta}


def asr_completed(item_id: str, transcript: str, language: str = "zh") -> dict[str, Any]:
    return {
        "type": ASR_COMPLETED,
        "item_id": item_id,
        "transcript": transcript,
        "language": language,
    }


def asr_failed(item_id: str, error: str) -> dict[str, Any]:
    return {"type": ASR_FAILED, "item_id": item_id, "error": error}


def _response_header(ctx: ResponseContext) -> dict[str, Any]:
    return {"response_id": ctx.response_id, "generation": ctx.generation}


def text_delta(ctx: ResponseContext, delta: str) -> dict[str, Any]:
    return {"type": TEXT_DELTA, **_response_header(ctx), "delta": delta}


def text_done(ctx: ResponseContext, text: str) -> dict[str, Any]:
    return {"type": TEXT_DONE, **_response_header(ctx), "text": text}


def audio_started(ctx: ResponseContext, rate: int = 24_000) -> dict[str, Any]:
    return {
        "type": AUDIO_STARTED,
        **_response_header(ctx),
        "format": {"type": "pcm_s16le", "rate": rate},
    }


def audio_delta(ctx: ResponseContext, delta_b64: str) -> dict[str, Any]:
    return {"type": AUDIO_DELTA, **_response_header(ctx), "delta": delta_b64}


def audio_done(ctx: ResponseContext) -> dict[str, Any]:
    return {"type": AUDIO_DONE, **_response_header(ctx)}


def tool_call_done(ctx: ResponseContext, calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "type": TOOL_CALL_DONE,
        **_response_header(ctx),
        "items": [
            {"call_id": call.call_id, "name": call.name, "arguments": call.arguments}
            for call in calls
        ],
    }


def response_done(ctx: ResponseContext, status: str = "completed") -> dict[str, Any]:
    return {"type": RESPONSE_DONE, **_response_header(ctx), "status": status}


def response_canceled(ctx: ResponseContext) -> dict[str, Any]:
    return {"type": RESPONSE_CANCELED, **_response_header(ctx)}


def backchannel(ctx: ResponseContext, sound_id: str, pcm: bytes, rate: int = 24_000) -> dict[str, Any]:
    """A short, separate filler clip.  It is not part of the formal response.

    The app plays it on a dedicated track and fades it out as soon as formal
    ``response.output_audio.started`` arrives, so it never delays the answer.
    """
    return {
        "type": BACKCHANNEL,
        **_response_header(ctx),
        "sound_id": sound_id,
        "format": {"type": "pcm_s16le", "rate": rate},
        "delta": base64.b64encode(pcm).decode("ascii"),
    }


def error(message: str, code: str = "gateway_error", fatal: bool = True) -> dict[str, Any]:
    return {
        "type": ERROR,
        "error": {"message": message, "code": code, "fatal": bool(fatal)},
    }


# ---------------------------------------------------------------------------
# Upstream parsers
# ---------------------------------------------------------------------------


def extract_tool_results(event: dict[str, Any]) -> list[ToolResult]:
    """Parse the app's conversation.item.create tool-result event.

    The app sends one or more items in a single event:
      {"type":"message","role":"tool","call_id":"...",
       "content":[{"type":"input_text","text":"..."}]}
    """
    results: list[ToolResult] = []
    items = event.get("items")
    if not isinstance(items, list):
        return results
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("role") != "tool":
            continue
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            continue
        text_parts: list[str] = []
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
        elif isinstance(content, str):
            text_parts.append(content)
        text = "".join(text_parts).strip() or "ok"
        results.append(ToolResult(call_id=call_id, text=text))
    return results


def extract_audio(event: dict[str, Any]) -> str:
    value = event.get("audio")
    if isinstance(value, str):
        return value
    # Tolerate a few alternate field names seen in Doubao examples, but the
    # Android app always sends "audio".
    for key in ("delta", "data"):
        alt = event.get(key)
        if isinstance(alt, str):
            return alt
    return ""


def extract_text(event: dict[str, Any]) -> str:
    value = event.get("text")
    return value if isinstance(value, str) else ""
