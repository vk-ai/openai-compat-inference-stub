"""OpenAI-ish request/response schemas for chat completions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | None = ""
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="stub-model")
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=256, ge=1)
    stream: bool = False
    n: int | None = Field(default=1, ge=1, le=1)
    stop: str | list[str] | None = None
    user: str | None = None
    # Extra OpenAI fields accepted but ignored by the stub
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI wire format


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    # Custom observability field (also mirrored in X-Latency-Ms header)
    latency_ms: float


class ChatCompletionChunkDeltaToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class ChatCompletionChunkDeltaToolCall(BaseModel):
    index: int = 0
    id: str | None = None
    type: Literal["function"] | None = None
    function: ChatCompletionChunkDeltaToolCallFunction | None = None


class ChatCompletionChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    tool_calls: list[ChatCompletionChunkDeltaToolCall] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionChunkDelta
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
