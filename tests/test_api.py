"""Offline API tests via FastAPI TestClient (httpx)."""

from __future__ import annotations

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
    metrics.total_latency_ms = 0.0
    metrics.last_latency_ms = 0.0
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
        json={"messages": [{"role": "user", "content": "alpha"}],
    ).json()
    b = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "beta"}]},
    ).json()
    assert a["choices"][0]["message"]["content"] != b["choices"][0]["message"]["content"]


def test_stream_rejected(client: TestClient):
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert r.status_code == 400
    assert "stream" in r.json()["detail"].lower()


def test_validation_requires_messages(client: TestClient):
    r = client.post("/v1/chat/completions", json={"model": "stub-model"})
    assert r.status_code == 422


def test_metrics_endpoint(client: TestClient):
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "m1"}]},
    )
    client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "m2"}]},
    )
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["request_count"] == 2
    assert body["error_count"] == 0
    assert body["avg_latency_ms"] >= 0
    assert "model" in body


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
