"""FastAPI OpenAI-compatible chat completions stub."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from app import __version__
from app.metrics import Timer, metrics
from app.mock_model import DEFAULT_MODEL_ID, generate
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
)

app = FastAPI(
    title="OpenAI-Compatible Inference Stub",
    description=(
        "OSS/learning FastAPI stub that implements a minimal OpenAI-compatible "
        "POST /v1/chat/completions surface with a deterministic mock model and "
        "latency metrics. Not employer production software."
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/metrics")
def get_metrics() -> dict[str, Any]:
    """Simple JSON metrics (request count + latency aggregates)."""
    snap = metrics.snapshot()
    snap["model"] = os.getenv("STUB_MODEL_ID", DEFAULT_MODEL_ID)
    return snap


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest, response: Response) -> ChatCompletionResponse:
    if body.stream:
        raise HTTPException(
            status_code=400,
            detail="stream=true is not supported by this stub; set stream=false",
        )

    timer = Timer()
    try:
        _artificial_delay()
        model_id = body.model or os.getenv("STUB_MODEL_ID", DEFAULT_MODEL_ID)
        result = generate(body.messages, model=model_id, max_tokens=body.max_tokens)
        latency_ms = round(timer.elapsed_ms(), 3)
        metrics.record(latency_ms, error=False)

        response.headers["X-Latency-Ms"] = str(latency_ms)
        response.headers["X-Stub-Model"] = model_id

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionMessage(content=result.content),
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
    except HTTPException:
        latency_ms = round(timer.elapsed_ms(), 3)
        metrics.record(latency_ms, error=True)
        response.headers["X-Latency-Ms"] = str(latency_ms)
        raise
    except Exception as exc:  # pragma: no cover - defensive
        latency_ms = round(timer.elapsed_ms(), 3)
        metrics.record(latency_ms, error=True)
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"error": {"message": str(exc), "type": "stub_error"}},
            headers={"X-Latency-Ms": str(latency_ms)},
        )
