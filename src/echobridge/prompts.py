"""System-prompt assembly for the LLM.

The device is the only persona source.  ``session.create.instructions`` is
forwarded verbatim; EchoBridge never supplies, defaults to, concatenates, or
caches a character prompt of its own.  The rules appended here are limited to
voice/output/tool constraints: they contain no character name, no
character-specific appearance, no role, and no catchphrase.  Rule 7 only tells
the model to stay consistent with whatever appearance the client supplied.

An empty instruction string gets a neutral minimal prompt.  It must never fall
back to a demo character.
"""

from __future__ import annotations

import hashlib

NEUTRAL_INSTRUCTIONS = "你是一个语音助手。"

VOICE_RULES = """【实时语音与输出规则（与人格无关）】
1. 每次回复 1–2 句，口语化，不超过 60 个汉字。
2. 不要 Markdown、列表、标题、代码块、括号动作、表情符号、舞台说明。
3. 不要提“专家/工具/照片/API/模型”；不要解释推理过程。
4. 需要看用户眼前画面时调用 look_and_ask；不要猜。
5. 不要调用 ask_expert（该工具在本后端不可用）；复杂问题直接简短回答或说明。
6. 句末要顺口，避免长串数字、英文缩写和生僻符号。
7. 回答涉及自身身体、外貌或穿着的问题时，以人格设定中的描述为准；设定里写了鞋、袜、衣物等穿戴，就按有相应的身体部位来回答，不要用现实物种的常识否认设定。"""


def build_system_prompt(instructions: str) -> str:
    """Return the exact system prompt sent to the LLM.

    ``instructions`` is preserved byte-for-byte (no strip, no truncation).
    Only a whitespace-only value is treated as empty and replaced by the
    neutral prompt.
    """
    text = instructions if isinstance(instructions, str) else ""
    base = text if text.strip() else NEUTRAL_INSTRUCTIONS
    return f"{base}\n\n{VOICE_RULES}"


def instructions_fingerprint(instructions: str) -> tuple[int, int, str]:
    """Return (chars, UTF-8 bytes, sha256 first 8) for session logging.

    The digest is over UTF-8 bytes so the Android client can reproduce it with
    ``MessageDigest.getInstance("SHA-256")`` over the same string.
    """
    text = instructions if isinstance(instructions, str) else ""
    encoded = text.encode("utf-8")
    return len(text), len(encoded), hashlib.sha256(encoded).hexdigest()[:8]
