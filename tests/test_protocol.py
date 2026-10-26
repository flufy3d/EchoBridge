from __future__ import annotations

import json

from echobridge import protocol


def test_session_created_shape():
    event = protocol.session_created("sess_x", "1.2.6.1")
    assert event["type"] == "session.created"
    assert event["session"]["id"] == "sess_x"


def test_audio_started_before_delta_contract():
    ctx = protocol.ResponseContext("resp_1", 1)
    started = protocol.audio_started(ctx)
    delta = protocol.audio_delta(ctx, "AAAA")
    assert started["response_id"] == "resp_1"
    assert delta["response_id"] == "resp_1"
    assert "delta" in delta


def test_tool_call_items_shape():
    ctx = protocol.ResponseContext("resp_2", 2)
    event = protocol.tool_call_done(
        ctx,
        [
            protocol.ToolCall("call_a", "set_emotion", '{"emotion":"happy"}'),
            protocol.ToolCall("call_b", "get_time", "{}"),
        ],
    )
    assert event["type"] == "response.function_call_arguments.done"
    assert [item["call_id"] for item in event["items"]] == ["call_a", "call_b"]


def test_tool_result_parser_accepts_multiple_items():
    event = {
        "type": "conversation.item.create",
        "items": [
            {
                "type": "message",
                "role": "tool",
                "call_id": "call_a",
                "content": [{"type": "input_text", "text": "ok"}],
            },
            {
                "type": "message",
                "role": "tool",
                "call_id": "call_b",
                "content": [{"type": "input_text", "text": "12:00"}],
            },
        ],
    }
    results = protocol.extract_tool_results(event)
    assert [(r.call_id, r.text) for r in results] == [("call_a", "ok"), ("call_b", "12:00")]


def test_parse_event_rejects_non_object():
    assert protocol.parse_event('{"type":"x"}')["type"] == "x"
    try:
        protocol.parse_event("[]")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_dumps_keeps_chinese_readable():
    raw = protocol.dumps({"type": "x", "text": "你好"})
    assert "你好" in raw
    assert json.loads(raw)["text"] == "你好"
