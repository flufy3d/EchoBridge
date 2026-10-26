from __future__ import annotations

from echobridge.tools import available_tool_names, convert_tools
from tests.helpers import ask_expert_tool, get_time_tool, set_emotion_tool


def test_convert_and_filter_ask_expert():
    tools = convert_tools([ask_expert_tool(), set_emotion_tool(), get_time_tool()])
    names = available_tool_names(tools)
    assert "ask_expert" not in names
    assert names == {"set_emotion", "get_time"}
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "set_emotion"


def test_convert_ignores_malformed_tools():
    tools = convert_tools([{"type": "function"}, None, {"name": "x", "parameters": "bad"}])
    assert len(tools) == 1
    assert tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}
