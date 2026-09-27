"""Deterministic mock model — no GPU, no network."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.schemas import ChatMessage, FunctionCall, ToolCall

DEFAULT_MODEL_ID = os.getenv("STUB_MODEL_ID", "stub-model")


@dataclass(frozen=True)
class MockResult:
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"
    tool_calls: list[ToolCall] = field(default_factory=list)


def _approx_tokens(text: str) -> int:
    """Rough whitespace/punctuation token estimate (good enough for a stub)."""
    if not text:
        return 0
    parts = re.findall(r"\S+", text)
    return max(1, len(parts))


def _last_user_content(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            return msg.content.strip()
    for msg in reversed(messages):
        if msg.content:
            return msg.content.strip()
    return ""


def _system_hint(messages: list[ChatMessage]) -> str:
    for msg in messages:
        if msg.role == "system" and msg.content:
            return msg.content.strip()[:120]
    return ""


def _tool_name_from_def(tool: dict[str, Any]) -> str | None:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        name = tool["function"].get("name")
        return str(name) if name else None
    # Tolerate bare {"name": ...} shapes some clients send in tests
    name = tool.get("name")
    return str(name) if name else None


def _select_tool(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> dict[str, Any] | None:
    """Pick which tool to mock-call, or None for a normal text completion."""
    if not tools:
        return None
    if tool_choice in (None, "auto"):
        return tools[0]
    if tool_choice == "none":
        return None
    if tool_choice == "required":
        return tools[0]
    if isinstance(tool_choice, dict):
        # {"type":"function","function":{"name":"..."}}
        wanted: str | None = None
        if tool_choice.get("type") == "function":
            fn = tool_choice.get("function") or {}
            wanted = fn.get("name")
        elif "name" in tool_choice:
            wanted = tool_choice.get("name")
        if wanted:
            for t in tools:
                if _tool_name_from_def(t) == wanted:
                    return t
        return tools[0]
    return tools[0]


def _mock_tool_arguments(tool: dict[str, Any], user_text: str, model: str) -> str:
    """Deterministic JSON-string arguments (OpenAI wire format)."""
    name = _tool_name_from_def(tool) or "unknown"
    digest = hashlib.sha256(f"{model}|{name}|{user_text}".encode()).hexdigest()[:10]
    # Prefer schema property names when present; fall back to query/input.
    props: dict[str, Any] = {}
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    params = fn.get("parameters") if isinstance(fn, dict) else None
    if isinstance(params, dict) and isinstance(params.get("properties"), dict):
        for key in params["properties"]:
            props[key] = user_text if key in {"query", "q", "input", "text", "prompt"} else user_text
            break
    if not props:
        props = {"query": user_text or "(empty)"}
    props["_stub_digest"] = digest
    return json.dumps(props, separators=(",", ":"), sort_keys=True)


def generate(
    messages: list[ChatMessage],
    *,
    model: str,
    max_tokens: int | None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
) -> MockResult:
    """
    Build a deterministic assistant reply from the conversation.

    When ``tools`` is present and ``tool_choice`` is not ``none``, emit a mock
    ``tool_calls`` payload (``finish_reason=tool_calls``) instead of text.
    Same inputs always produce the same content / tool_calls (no randomness).
    """
    user_text = _last_user_content(messages)
    system = _system_hint(messages)
    prompt_text = "\n".join(f"{m.role}: {m.content or ''}" for m in messages)
    prompt_tokens = _approx_tokens(prompt_text)

    selected = _select_tool(tools, tool_choice)
    if selected is not None:
        name = _tool_name_from_def(selected) or "tool"
        args_json = _mock_tool_arguments(selected, user_text, model)
        call_id = "call_" + hashlib.sha256(f"{model}|{name}|{args_json}".encode()).hexdigest()[:24]
        tool_call = ToolCall(
            id=call_id,
            type="function",
            function=FunctionCall(name=name, arguments=args_json),
        )
        completion_tokens = _approx_tokens(name + " " + args_json)
        return MockResult(
            content=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason="tool_calls",
            tool_calls=[tool_call],
        )

    digest = hashlib.sha256(
        f"{model}|{system}|{user_text}|{''.join((m.role + ':' + (m.content or '')) for m in messages)}".encode()
    ).hexdigest()[:12]

    lines = [
        f"[stub:{model}] deterministic reply",
        f"digest={digest}",
    ]
    if system:
        lines.append(f"system_hint={system}")
    if user_text:
        lines.append(f"echo={user_text}")
    else:
        lines.append("echo=(empty)")

    content = "\n".join(lines)
    if max_tokens is not None and max_tokens > 0:
        words = content.split()
        if len(words) > max_tokens:
            content = " ".join(words[:max_tokens])
            finish = "length"
        else:
            finish = "stop"
    else:
        finish = "stop"

    completion_tokens = _approx_tokens(content)
    return MockResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish,
        tool_calls=[],
    )
