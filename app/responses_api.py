"""Minimal, stateless ``POST /v1/responses`` (OpenAI Responses API shape).

Newer clients (Codex CLI, recent SDKs, agent frameworks) talk to the Responses
API instead of Chat Completions. This module maps a Responses request onto the
same deterministic mock model used by ``/v1/chat/completions`` and renders:

- **non-stream**: a ``response`` object whose ``output[]`` holds a ``message``
  item (``output_text`` content) or a ``function_call`` item
- **stream**: typed SSE events (``event: <type>`` + JSON ``data`` with a
  ``sequence_number``): ``response.created`` → ``response.in_progress`` →
  ``response.output_item.added`` → ``response.content_part.added`` →
  ``response.output_text.delta``* → ``response.output_text.done`` →
  ``response.content_part.done`` → ``response.output_item.done`` →
  ``response.completed`` (or ``response.incomplete`` on ``max_output_tokens``).
  Function calls use ``response.function_call_arguments.delta`` / ``.done``.
  There is no ``[DONE]`` sentinel in this API.

**Stateless:** nothing is stored. ``previous_response_id`` is rejected with a
400 telling the client to resend the full conversation in ``input``, and
responses report ``store: false``. IDs are derived from a hash of the request,
so the same request always gets the same ids and output (``created_at`` is
wall-clock time).

Supported ``input`` items: a plain string; ``{"role": ..., "content": str |
[{"type": "input_text"|"output_text"|"text", "text": ...}]}`` messages
(``developer`` is treated as ``system``); ``function_call`` and
``function_call_output`` items for tool round-trips. Only ``type: function``
tools are used; other tool types are ignored and listed in the
``X-Stub-Ignored-Tools`` header.

Deterministic tool rule: if the last input item is a ``function_call_output``,
the stub answers with text (unless ``tool_choice`` is ``required`` or names a
function), so agent loops terminate.

OSS/learning only: a subset of the Responses API, not OpenAI parity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, Field

from app.mock_model import MockResult
from app.schemas import ChatMessage


class ResponsesRequest(BaseModel):
    model: str = Field(default="stub-model")
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    stream: bool = False
    max_output_tokens: int | None = Field(default=None, ge=1)
    previous_response_id: str | None = None
    store: bool | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = None
    metadata: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    user: str | None = None


class ResponsesInputError(ValueError):
    def __init__(self, message: str, param: str = "input", code: str = "invalid_input") -> None:
        super().__init__(message)
        self.param = param
        self.code = code


def error_payload(message: str, *, param: str | None, code: str) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }
    }


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                ptype = part.get("type")
                if ptype in ("input_text", "output_text", "text"):
                    parts.append(str(part.get("text", "")))
                elif ptype in ("input_image", "input_file"):
                    parts.append(f"[{ptype}]")
                else:
                    raise ResponsesInputError(f"unsupported content part type {ptype!r}")
        return "".join(parts)
    raise ResponsesInputError("message content must be a string or a list of content parts")


def to_chat_messages(req: ResponsesRequest) -> list[ChatMessage]:
    """Responses ``instructions`` + ``input`` → chat-style messages for the mock model."""
    msgs: list[ChatMessage] = []
    if req.instructions:
        msgs.append(ChatMessage(role="system", content=req.instructions))
    if isinstance(req.input, str):
        msgs.append(ChatMessage(role="user", content=req.input))
        return msgs
    if not req.input:
        raise ResponsesInputError("input must not be empty")
    for item in req.input:
        itype = item.get("type")
        if itype in (None, "message") and "role" in item:
            role = item["role"]
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant"):
                raise ResponsesInputError(f"unsupported message role {item['role']!r}")
            msgs.append(ChatMessage(role=role, content=_content_text(item.get("content"))))
        elif itype == "function_call":
            msgs.append(
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": item.get("call_id"),
                            "type": "function",
                            "function": {
                                "name": item.get("name"),
                                "arguments": item.get("arguments", ""),
                            },
                        }
                    ],
                )
            )
        elif itype == "function_call_output":
            output = item.get("output", "")
            if not isinstance(output, str):
                output = json.dumps(output, sort_keys=True)
            msgs.append(ChatMessage(role="tool", content=output, tool_call_id=item.get("call_id")))
        elif itype in ("reasoning", "item_reference"):
            continue  # tolerated, carries nothing for a mock model
        else:
            raise ResponsesInputError(f"unsupported input item type {itype!r}")
    if not msgs:
        raise ResponsesInputError("input has no usable items")
    return msgs


def to_chat_tools(
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]] | None, list[str]]:
    """Flat Responses function tools → chat ``{"type":"function","function":{...}}``."""
    if not tools:
        return None, []
    chat: list[dict[str, Any]] = []
    ignored: list[str] = []
    for t in tools:
        if t.get("type") == "function" and t.get("name"):
            fn = {k: t[k] for k in ("name", "description", "parameters") if k in t}
            chat.append({"type": "function", "function": fn})
        else:
            ignored.append(str(t.get("type") or "unknown"))
    return (chat or None), ignored


def to_chat_tool_choice(choice: Any, input_items: str | list[dict[str, Any]]) -> Any:
    """Map Responses ``tool_choice``; answer in text right after a tool result."""
    if isinstance(choice, dict) and choice.get("type") == "function" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    last_is_tool_output = (
        isinstance(input_items, list)
        and bool(input_items)
        and input_items[-1].get("type") == "function_call_output"
    )
    if last_is_tool_output and choice in (None, "auto"):
        return "none"
    return choice


def response_ids(req: ResponsesRequest, model_id: str) -> tuple[str, str]:
    """Deterministic ``(resp_id, item_id_suffix)`` from the request content."""
    blob = json.dumps(
        {
            "model": model_id,
            "input": req.input,
            "instructions": req.instructions,
            "tools": req.tools,
            "tool_choice": req.tool_choice,
            "max_output_tokens": req.max_output_tokens,
        },
        sort_keys=True,
        default=str,
    )
    h = hashlib.sha256(blob.encode()).hexdigest()
    return f"resp_{h[:32]}", h[32:56]


def output_items(result: MockResult, suffix: str) -> list[dict[str, Any]]:
    if result.tool_calls:
        return [
            {
                "type": "function_call",
                "id": f"fc_{suffix}{i}",
                "call_id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "status": "completed",
            }
            for i, tc in enumerate(result.tool_calls)
        ]
    return [
        {
            "type": "message",
            "id": f"msg_{suffix}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": result.content or "", "annotations": []}],
        }
    ]


def build_response(
    req: ResponsesRequest,
    *,
    result: MockResult | None,
    model_id: str,
    resp_id: str,
    suffix: str,
    created_at: int,
    status: str | None = None,
) -> dict[str, Any]:
    """Full ``response`` object. ``result=None`` renders the in-progress shell."""
    incomplete = result is not None and result.finish_reason == "length"
    if status is None:
        status = "incomplete" if incomplete else "completed"
    usage = None
    if result is not None:
        usage = {
            "input_tokens": result.prompt_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": result.completion_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        }
    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": {"reason": "max_output_tokens"} if incomplete else None,
        "instructions": req.instructions,
        "max_output_tokens": req.max_output_tokens,
        "model": model_id,
        "output": output_items(result, suffix) if result is not None else [],
        "parallel_tool_calls": True if req.parallel_tool_calls is None else req.parallel_tool_calls,
        "previous_response_id": None,
        "store": False,  # stateless stub: nothing is persisted
        "temperature": req.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": req.tool_choice if req.tool_choice is not None else "auto",
        "tools": req.tools or [],
        "top_p": req.top_p,
        "usage": usage,
        "metadata": req.metadata or {},
    }


def _chunks(text: str, size: int) -> list[str]:
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def iter_events(
    req: ResponsesRequest,
    *,
    result: MockResult,
    model_id: str,
    resp_id: str,
    suffix: str,
    created_at: int,
    drop_after: int | None = None,
) -> Iterator[bytes]:
    """Typed Responses SSE events. ``drop_after=k`` stops after k delta events."""
    seq = 0
    deltas = 0

    def ev(payload: dict[str, Any]) -> bytes:
        nonlocal seq
        payload = {"type": payload["type"], "sequence_number": seq, **payload}
        seq += 1
        return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")

    shell = build_response(
        req, result=None, model_id=model_id, resp_id=resp_id, suffix=suffix,
        created_at=created_at, status="in_progress",
    )
    yield ev({"type": "response.created", "response": shell})
    yield ev({"type": "response.in_progress", "response": shell})

    for out_idx, item in enumerate(output_items(result, suffix)):
        if item["type"] == "function_call":
            yield ev({
                "type": "response.output_item.added",
                "output_index": out_idx,
                "item": {**item, "arguments": "", "status": "in_progress"},
            })
            for piece in _chunks(item["arguments"], 16):
                if drop_after is not None and deltas >= drop_after:
                    return
                deltas += 1
                yield ev({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": out_idx,
                    "delta": piece,
                })
            yield ev({
                "type": "response.function_call_arguments.done",
                "item_id": item["id"],
                "output_index": out_idx,
                "arguments": item["arguments"],
            })
            yield ev({"type": "response.output_item.done", "output_index": out_idx, "item": item})
            continue

        text = item["content"][0]["text"]
        yield ev({
            "type": "response.output_item.added",
            "output_index": out_idx,
            "item": {**item, "status": "in_progress", "content": []},
        })
        empty_part = {"type": "output_text", "text": "", "annotations": []}
        yield ev({
            "type": "response.content_part.added",
            "item_id": item["id"],
            "output_index": out_idx,
            "content_index": 0,
            "part": empty_part,
        })
        for piece in _chunks(text, 12):
            if drop_after is not None and deltas >= drop_after:
                return
            deltas += 1
            yield ev({
                "type": "response.output_text.delta",
                "item_id": item["id"],
                "output_index": out_idx,
                "content_index": 0,
                "delta": piece,
            })
        yield ev({
            "type": "response.output_text.done",
            "item_id": item["id"],
            "output_index": out_idx,
            "content_index": 0,
            "text": text,
        })
        yield ev({
            "type": "response.content_part.done",
            "item_id": item["id"],
            "output_index": out_idx,
            "content_index": 0,
            "part": item["content"][0],
        })
        yield ev({"type": "response.output_item.done", "output_index": out_idx, "item": item})

    if drop_after is not None:
        return  # drop:k larger than the delta count still never sends the terminal event
    final = build_response(
        req, result=result, model_id=model_id, resp_id=resp_id, suffix=suffix, created_at=created_at
    )
    terminal = "response.incomplete" if final["status"] == "incomplete" else "response.completed"
    yield ev({"type": terminal, "response": final})
