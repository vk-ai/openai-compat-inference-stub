"""FastAPI OpenAI-compatible chat completions stub."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app import __version__
from app.metrics import Timer, metrics
from app.mock_model import DEFAULT_MODEL_ID, MockResult, generate
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatCompletionChunkDeltaToolCall,
    ChatCompletionChunkDeltaToolCallFunction,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)

app = FastAPI(
    title="OpenAI-Compatible Inference Stub",
    description=(
        "OSS/learning FastAPI stub that implements a minimal OpenAI-compatible "
        "POST /v1/chat/completions surface (JSON + stream=true SSE) with a "
        "deterministic mock model, optional tools/tool_calls, latency, TTFT, "
        "and Prometheus /metrics. Not employer production software."
    ),
    version=__version__,
)


def _artificial_delay() -> None:
    raw = os.getenv("STUB_LATENCY_MS", "0")
    try:
        ms = float(raw)
    except ValueError:
        ms = 0.0
    if ms > 0:
        time.sleep(ms / 1000.0)


def _chunk_content(content: str) -> list[str]:
    """Split assistant text into small deterministic SSE content pieces."""
    if not content:
        return [""]
    if "\n" in content:
        parts = content.split("\n")
        out: list[str] = []
        for i, part in enumerate(parts):
            piece = part if i == len(parts) - 1 else part + "\n"
            if piece:
                out.append(piece)
        return out or [content]
    size = 12
    return [content[i : i + size] for i in range(0, len(content), size)]


def _chunk_tool_arguments(arguments: str, size: int = 16) -> list[str]:
    if not arguments:
        return [""]
    return [arguments[i : i + size] for i in range(0, len(arguments), size)]


def _iter_sse(
    *,
    result: MockResult,
    model_id: str,
    completion_id: str,
    created: int,
) -> Iterator[bytes]:
    if result.tool_calls:
        # Streaming tool_calls: role+id+name first, then argument deltas, then finish.
        # Matches the wire shape LangChain / GPTMock clients break on when wrong.
        for tc_index, tc in enumerate(result.tool_calls):
            first = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(
                            role="assistant" if tc_index == 0 else None,
                            tool_calls=[
                                ChatCompletionChunkDeltaToolCall(
                                    index=tc_index,
                                    id=tc.id,
                                    type="function",
                                    function=ChatCompletionChunkDeltaToolCallFunction(
                                        name=tc.function.name,
                                        arguments="",
                                    ),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {first.model_dump_json()}\n\n".encode("utf-8")

            for piece in _chunk_tool_arguments(tc.function.arguments):
                chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_id,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatCompletionChunkDelta(
                                tool_calls=[
                                    ChatCompletionChunkDeltaToolCall(
                                        index=tc_index,
                                        function=ChatCompletionChunkDeltaToolCallFunction(
                                            arguments=piece,
                                        ),
                                    )
                                ],
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")

        final = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(),
                    finish_reason="tool_calls",
                )
            ],
        )
        yield f"data: {final.model_dump_json()}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        return

    pieces = _chunk_content(result.content or "")
    first = pieces[0] if pieces else ""
    rest = pieces[1:] if len(pieces) > 1 else []

    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(role="assistant", content=first),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n".encode("utf-8")

    for piece in rest:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=ChatCompletionChunkDelta(content=piece),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n".encode("utf-8")

    final = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=ChatCompletionChunkDelta(),
                finish_reason=result.finish_reason,  # type: ignore[arg-type]
            )
        ],
    )
    yield f"data: {final.model_dump_json()}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/metrics")
def get_metrics_prometheus() -> Response:
    """Prometheus text exposition (scrapers). Content-Type: text/plain; version=0.0.4."""
    body = metrics.prometheus_text(model=os.getenv("STUB_MODEL_ID", DEFAULT_MODEL_ID))
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/metrics.json")
def get_metrics_json() -> dict[str, Any]:
    """JSON metrics (humans / simple dashboards). Same aggregates as /metrics."""
    snap = metrics.snapshot()
    snap["model"] = os.getenv("STUB_MODEL_ID", DEFAULT_MODEL_ID)
    return snap


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest, response: Response):
    timer = Timer()
    model_id = body.model or os.getenv("STUB_MODEL_ID", DEFAULT_MODEL_ID)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        _artificial_delay()
        result = generate(
            body.messages,
            model=model_id,
            max_tokens=body.max_tokens,
            tools=body.tools,
            tool_choice=body.tool_choice,
        )

        if body.stream:
            # TTFT = time until first SSE byte is ready (after mock "prefill")
            ttft_ms = round(timer.elapsed_ms(), 3)

            def stream_and_record() -> Iterator[bytes]:
                try:
                    yield from _iter_sse(
                        result=result,
                        model_id=model_id,
                        completion_id=completion_id,
                        created=created,
                    )
                finally:
                    latency_ms = round(timer.elapsed_ms(), 3)
                    metrics.record(
                        latency_ms, error=False, stream=True, ttft_ms=ttft_ms
                    )

            return StreamingResponse(
                stream_and_record(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Stub-Model": model_id,
                    "X-TTFT-Ms": str(ttft_ms),
                    "X-Latency-Ms": str(ttft_ms),  # end-to-end filled after stream; TTFT is primary
                },
            )

        latency_ms = round(timer.elapsed_ms(), 3)
        # Non-stream: first token == full JSON body
        metrics.record(latency_ms, error=False, stream=False, ttft_ms=latency_ms)

        response.headers["X-Latency-Ms"] = str(latency_ms)
        response.headers["X-TTFT-Ms"] = str(latency_ms)
        response.headers["X-Stub-Model"] = model_id

        message = ChatCompletionMessage(
            content=result.content,
            tool_calls=list(result.tool_calls) if result.tool_calls else None,
        )
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=message,
                    finish_reason=result.finish_reason,  # type: ignore[arg-type]
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
            latency_ms=latency_ms,
        )
    except Exception as exc:  # pragma: no cover - defensive
        latency_ms = round(timer.elapsed_ms(), 3)
        metrics.record(latency_ms, error=True)
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"error": {"message": str(exc), "type": "stub_error"}},
            headers={"X-Latency-Ms": str(latency_ms)},
        )
