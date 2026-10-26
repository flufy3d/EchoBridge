"""Tool schema conversion and filtering.

The app still sends the full Doubao/Volc tool list.  For the EchoBridge backend
the main model is already DeepSeek, so `ask_expert` is deliberately removed.
Everything else is forwarded unchanged.
"""

from __future__ import annotations

import json
from typing import Any

FILTERED_TOOL_NAMES = {"ask_expert"}


def convert_tools(volc_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(volc_tools, list):
        return []
    result: list[dict[str, Any]] = []
    for item in volc_tools:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in FILTERED_TOOL_NAMES:
            continue
        function: dict[str, Any] = {
            "name": name,
            "description": item.get("description", ""),
            "parameters": item.get("parameters")
            if isinstance(item.get("parameters"), dict)
            else {"type": "object", "properties": {}},
        }
        result.append({"type": "function", "function": function})
    return result


def available_tool_names(openai_tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in openai_tools:
        function = item.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def json_arguments(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
