"""Offline API tests via FastAPI TestClient (httpx)."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

# Ensure deterministic env before app import side effects
os.environ.pop("STUB_LATENCY_MS", None)
os.environ["STUB_MODEL_ID"] = "stub-model"

from app.main import app  # noqa: E402
from app.metrics import metrics  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.request_count = 0
    metrics.error_count = 0
    metrics.stream_request_count = 0
    metrics.total_latency_ms = 0.0
    metrics.last_latency_ms = 0.0
    metrics.total_ttft_ms = 0.0
    metrics.last_ttft_ms = 0.0
    metrics.ttft_sample_count = 0
    yield


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_chat_completions_shape(client: TestClient):
    payload = {
        "model": "stub-model",
        "messages": [
            {"role": "system", "content": "You are a stub."},
            {"role": "user", "content": "Hello stub"},
        ],
        "stream": False,
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    assert "X-Latency-Ms" in r.headers
    assert "X-TTFT-Ms" in r.headers
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "stub-model"
    assert data["id"].startswith("chatcmpl-")
    assert len(data["choices"]) == 1
    msg = data["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "Hello stub" in msg["content"]
    assert "digest=" in msg["content"]
    assert data["usage"]["total_tokens"] >= data["usage"]["prompt_tokens"]
    assert isinstance(data["latency_ms"], (int, float))
    assert data["latency_ms"] >= 0


def test_deterministic_same_input(client: TestClient):
    payload = {
        "messages": [{"role": "user", "content": "ping"}],
    }
    a = client.post("/v1/chat/completions", json=payload).json()
    b = client.post("/v1/chat/completions", json=payload).json()
    assert a["choices"][0]["message"]["content"] == b["choices"][0]["message"]["content"]


def test_different_input_different_digest(client: TestClient):
    a = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "alpha"}]},
    ).json()
    b = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "beta"}]},
    ).json()
    assert a["choices"][0]["message"]["content"] != b["choices"][0]["message"]["content"]


def test_stream_sse_openai_wire(client: TestClient):
    """stream=true must emit OpenAI-compatible SSE chunks ending with [DONE]."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        assert "X-TTFT-Ms" in r.headers
        assert float(r.headers["X-TTFT-Ms"]) >= 0
        raw = r.read().decode("utf-8")

    assert "data: [DONE]" in raw
    events = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            events.append("[DONE]")
            continue
        events.append(json.loads(payload))

    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert len(chunks) >= 2
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    assembled = "".join(
        (c["choices"][0]["delta"].get("content") or "") for c in chunks
    )
    assert "stream please" in assembled
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")


def test_stream_matches_non_stream_content(client: TestClient):
    payload = {"messages": [{"role": "user", "content": "parity check"}]}
    non = client.post("/v1/chat/completions", json={**payload, "stream": False}).json()
    expected = non["choices"][0]["message"]["content"]

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={**payload, "stream": True},
    ) as r:
        raw = r.read().decode("utf-8")

    parts: list[str] = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        if data.strip() == "[DONE]":
            break
        chunk = json.loads(data)
        parts.append(chunk["choices"][0]["delta"].get("content") or "")
    assert "".join(parts) == expected


def test_validation_requires_messages(client: TestClient):
    r = client.post("/v1/chat/completions", json={"model": "stub-model"})
    assert r.status_code == 422


def test_metrics_json_endpoint(client: TestClient):
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "m1"}]},
    )
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "m2"}], "stream": True},
    ) as r:
        r.read()
    r = client.get("/metrics.json")
    assert r.status_code == 200
    body = r.json()
    assert body["request_count"] == 2
    assert body["stream_request_count"] == 1
    assert body["error_count"] == 0
    assert body["avg_latency_ms"] >= 0
    assert body["ttft_sample_count"] == 2
    assert body["avg_ttft_ms"] >= 0
    assert "last_ttft_ms" in body
    assert "model" in body


def test_metrics_prometheus_text(client: TestClient):
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "prom"}]},
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    ctype = r.headers.get("content-type", "")
    assert "text/plain" in ctype
    assert "version=0.0.4" in ctype
    body = r.text
    assert "# HELP stub_requests_total" in body
    assert "# TYPE stub_requests_total counter" in body
    assert "stub_requests_total{" in body
    assert "stub_ttft_ms_avg{" in body
    assert "stub_stream_requests_total{" in body
    # At least one sample line with a numeric value
    assert any(
        line.startswith("stub_requests_total") and line.rstrip().endswith("1")
        for line in body.splitlines()
    )


def test_max_tokens_truncation(client: TestClient):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "truncate me please with extra words"}],
            "max_tokens": 3,
        },
    )
    assert r.status_code == 200
    data = r.json()
    words = data["choices"][0]["message"]["content"].split()
    assert len(words) <= 3
    assert data["choices"][0]["finish_reason"] == "length"


_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Mock search",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def test_tool_calls_non_stream(client: TestClient):
    payload = {
        "model": "stub-model",
        "messages": [{"role": "user", "content": "find Paris"}],
        "tools": [_SEARCH_TOOL],
        "tool_choice": "auto",
    }
    r = client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    data = r.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert msg["content"] is None
    assert msg["tool_calls"] and len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"].startswith("call_")
    assert tc["function"]["name"] == "search"
    args = json.loads(tc["function"]["arguments"])
    assert args["query"] == "find Paris"
    assert "_stub_digest" in args


def test_tool_calls_deterministic(client: TestClient):
    payload = {
        "messages": [{"role": "user", "content": "same tools"}],
        "tools": [_SEARCH_TOOL],
    }
    a = client.post("/v1/chat/completions", json=payload).json()
    b = client.post("/v1/chat/completions", json=payload).json()
    assert a["choices"][0]["message"]["tool_calls"] == b["choices"][0]["message"]["tool_calls"]


def test_tool_choice_none_skips_tools(client: TestClient):
    payload = {
        "messages": [{"role": "user", "content": "no tools please"}],
        "tools": [_SEARCH_TOOL],
        "tool_choice": "none",
    }
    data = client.post("/v1/chat/completions", json=payload).json()
    choice = data["choices"][0]
    assert choice["finish_reason"] in ("stop", "length")
    assert choice["message"].get("tool_calls") in (None, [])
    assert "no tools please" in (choice["message"]["content"] or "")


def test_tool_calls_stream_deltas(client: TestClient):
    """Streaming must emit stable-index tool_call chunks + finish_reason tool_calls."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "stream tools"}],
            "tools": [_SEARCH_TOOL],
            "stream": True,
        },
    ) as r:
        assert r.status_code == 200
        raw = r.read().decode("utf-8")

    assert "data: [DONE]" in raw
    chunks = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            break
        chunks.append(json.loads(payload))

    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    # First delta carries role + tool call id/name
    first_delta = chunks[0]["choices"][0]["delta"]
    assert first_delta.get("role") == "assistant"
    assert first_delta["tool_calls"][0]["id"].startswith("call_")
    assert first_delta["tool_calls"][0]["function"]["name"] == "search"
    assert first_delta["tool_calls"][0]["index"] == 0

    # Argument pieces reassemble to valid JSON
    arg_parts = []
    for c in chunks:
        delta = c["choices"][0]["delta"]
        for tc in delta.get("tool_calls") or []:
            assert tc.get("index", 0) == 0
            fn = tc.get("function") or {}
            if fn.get("arguments"):
                arg_parts.append(fn["arguments"])
    assembled = "".join(arg_parts)
    parsed = json.loads(assembled)
    assert parsed["query"] == "stream tools"
