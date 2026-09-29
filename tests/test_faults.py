"""Deterministic fault injection: 429 + Retry-After, 503, timeout, mid-stream drop."""

from __future__ import annotations

import json
import os
import time

import pytest
from fastapi.testclient import TestClient

os.environ.pop("STUB_LATENCY_MS", None)
os.environ["STUB_MODEL_ID"] = "stub-model"

from app.faults import (  # noqa: E402
    FaultSpecError,
    RateLimitWindow,
    faults,
    fmt_retry_after,
    parse_fault_spec,
    rate_limits,
    split_model_fault,
)
from app.main import app  # noqa: E402

FAULT_ENV = ("STUB_FAULT", "STUB_FAULT_FIRST", "STUB_FAULT_EVERY", "STUB_FAULT_TIMEOUT_MS",
             "STUB_RATELIMIT_REQUESTS", "STUB_RATELIMIT_TOKENS")
PAYLOAD = {"messages": [{"role": "user", "content": "hello faults"}]}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in FAULT_ENV:
        monkeypatch.delenv(k, raising=False)
    faults.reset()
    rate_limits.reset()
    yield
    faults.reset()
    rate_limits.reset()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _sse_lines(resp) -> list[str]:
    return [ln for ln in resp.iter_lines() if ln.startswith("data: ")]


# --- parsing -----------------------------------------------------------------

def test_parse_fault_spec():
    assert parse_fault_spec("429").retry_after_s == 1.0
    assert parse_fault_spec("429:7").retry_after_s == 7.0
    assert parse_fault_spec("503").retry_after_s is None
    assert parse_fault_spec("timeout:250").timeout_ms == 250.0
    assert parse_fault_spec("drop:3").drop_after == 3
    assert parse_fault_spec("drop").drop_after == 2
    assert parse_fault_spec("none") is None and parse_fault_spec("") is None
    with pytest.raises(FaultSpecError):
        parse_fault_spec("418")


def test_split_model_fault_and_retry_after_rounding():
    assert split_model_fault("stub-model::fault=503") == ("stub-model", "503")
    assert split_model_fault("stub-model") == ("stub-model", None)
    assert fmt_retry_after(2.0) == "2"
    assert fmt_retry_after(0.2) == "1"  # round up: never invite an early retry


# --- default off + always-on headers --------------------------------------

def test_off_by_default_and_ratelimit_headers_present(client):
    r = client.post("/v1/chat/completions", json=PAYLOAD)
    assert r.status_code == 200
    assert "X-Stub-Fault" not in r.headers
    for h in ("limit-requests", "remaining-requests", "reset-requests",
              "limit-tokens", "remaining-tokens", "reset-tokens"):
        assert f"x-ratelimit-{h}" in r.headers
    assert int(r.headers["x-ratelimit-remaining-requests"]) == 59
    r2 = client.post("/v1/chat/completions", json=PAYLOAD)
    assert int(r2.headers["x-ratelimit-remaining-requests"]) == 58


def test_ratelimit_window_resets_with_fake_clock():
    now = [0.0]
    w = RateLimitWindow(clock=lambda: now[0])
    assert w.headers()["x-ratelimit-remaining-requests"] == "59"
    now[0] = 30.0
    h = w.headers()
    assert h["x-ratelimit-remaining-requests"] == "58"
    assert h["x-ratelimit-reset-requests"] == "30s"
    now[0] = 61.0
    assert w.headers()["x-ratelimit-remaining-requests"] == "59"


# --- 429 / 503 ---------------------------------------------------------------

def test_header_429_openai_shape_with_retry_after(client):
    r = client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "429:3"})
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "3"
    assert r.headers["X-Stub-Fault"] == "429"
    assert r.headers["x-ratelimit-remaining-requests"] == "0"
    assert r.headers["x-ratelimit-reset-requests"] == "3s"
    err = r.json()["error"]
    assert err["code"] == "rate_limit_exceeded"
    assert err["type"] == "requests"


def test_model_suffix_503(client):
    r = client.post(
        "/v1/chat/completions",
        json={**PAYLOAD, "model": "stub-model::fault=503:2"},
    )
    assert r.status_code == 503
    assert r.headers["Retry-After"] == "2"
    assert r.headers["X-Stub-Model"] == "stub-model"  # suffix stripped
    assert r.json()["error"]["code"] == "service_unavailable"


def test_503_without_retry_after(client):
    r = client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "503"})
    assert r.status_code == 503
    assert "Retry-After" not in r.headers


def test_invalid_fault_spec_is_400(client):
    r = client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "teapot"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_fault_spec"


# --- env schedule (deterministic counters) -----------------------------------

def test_env_first_n_then_healthy(client, monkeypatch):
    """Retry-then-succeed shape: first 2 requests 429, then 200."""
    monkeypatch.setenv("STUB_FAULT", "429:1")
    monkeypatch.setenv("STUB_FAULT_FIRST", "2")
    codes = [client.post("/v1/chat/completions", json=PAYLOAD).status_code for _ in range(4)]
    assert codes == [429, 429, 200, 200]


def test_env_every_n(client, monkeypatch):
    monkeypatch.setenv("STUB_FAULT", "503")
    monkeypatch.setenv("STUB_FAULT_EVERY", "3")
    codes = [client.post("/v1/chat/completions", json=PAYLOAD).status_code for _ in range(6)]
    assert codes == [200, 200, 503, 200, 200, 503]


def test_header_none_overrides_env(client, monkeypatch):
    monkeypatch.setenv("STUB_FAULT", "429")
    assert client.post("/v1/chat/completions", json=PAYLOAD).status_code == 429
    r = client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "none"})
    assert r.status_code == 200


# --- timeout -----------------------------------------------------------------

def test_timeout_delays_then_answers(client):
    t0 = time.perf_counter()
    r = client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "timeout:150"})
    elapsed = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200
    assert elapsed >= 140
    assert r.headers["X-Stub-Fault"] == "timeout"


# --- mid-stream drop -----------------------------------------------------------

def test_drop_mid_stream_no_done_no_finish_reason(client):
    long_msg = {"messages": [{"role": "user", "content": "please stream a fairly long reply " * 3}],
                "stream": True}
    with client.stream("POST", "/v1/chat/completions", json=long_msg,
                       headers={"X-Stub-Fault": "drop:2"}) as r:
        assert r.status_code == 200
        assert r.headers["X-Stub-Fault"] == "drop"
        lines = _sse_lines(r)
    assert len(lines) == 2
    assert "data: [DONE]" not in lines
    for ln in lines:
        chunk = json.loads(ln[len("data: "):])
        assert chunk["choices"][0]["finish_reason"] is None

    # Same request without the fault: more chunks, finish_reason, and [DONE]
    with client.stream("POST", "/v1/chat/completions", json=long_msg) as r:
        healthy = _sse_lines(r)
    assert healthy[-1] == "data: [DONE]"
    assert len(healthy) > 4


def test_drop_ignored_for_non_stream(client):
    r = client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "drop:1"})
    assert r.status_code == 200
    assert "X-Stub-Fault" not in r.headers


# --- metrics -----------------------------------------------------------------

def test_faults_counted_in_metrics(client):
    client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "429"})
    client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "429"})
    client.post("/v1/chat/completions", json=PAYLOAD, headers={"X-Stub-Fault": "503"})
    snap = client.get("/metrics.json").json()
    assert snap["faults_injected"] == {"429": 2, "503": 1}
    text = client.get("/metrics").text
    assert 'stub_faults_injected_total{model="stub-model",kind="429"} 2' in text
    assert 'stub_faults_injected_total{model="stub-model",kind="drop"} 0' in text
