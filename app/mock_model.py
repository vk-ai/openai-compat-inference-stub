"""Deterministic mock model — no GPU, no network."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from app.schemas import ChatMessage

DEFAULT_MODEL_ID = os.getenv("STUB_MODEL_ID", "stub-model")


@dataclass(frozen=True)
class MockResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"


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
    # Fall back to last non-empty content
    for msg in reversed(messages):
        if msg.content:
            return msg.content.strip()
    return ""


def _system_hint(messages: list[ChatMessage]) -> str:
    for msg in messages:
        if msg.role == "system" and msg.content:
            return msg.content.strip()[:120]
    return ""


def generate(messages: list[ChatMessage], *, model: str, max_tokens: int | None) -> MockResult:
    """
    Build a deterministic assistant reply from the conversation.

    Same messages + model always produce the same content (no randomness).
    """
    user_text = _last_user_content(messages)
    system = _system_hint(messages)
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
        # Soft truncate by approximate tokens (word units)
        words = content.split()
        if len(words) > max_tokens:
            content = " ".join(words[:max_tokens])
            finish = "length"
        else:
            finish = "stop"
    else:
        finish = "stop"

    prompt_text = "\n".join(f"{m.role}: {m.content or ''}" for m in messages)
    prompt_tokens = _approx_tokens(prompt_text)
    completion_tokens = _approx_tokens(content)
    return MockResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish,
    )
