"""OpenAI-ish request/response schemas for chat completions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = "user"
    content: str | None = ""
    name: str | None = None


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


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length"] = "stop"


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
