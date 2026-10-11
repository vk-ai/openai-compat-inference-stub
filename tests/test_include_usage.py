"""stream_options.include_usage: final usage chunk (OpenAI wire shape)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "include_usage_stream.json").read_text())


@pytest.fixture()
def client():
    return TestClient(app)


def _events(raw: str) -> list:
    out = []
    for line in raw.splitlines():
        if line.startswith("data: "):
            p = line[6:]
            out.append("[DONE]" if p.strip() == "[DONE]" else json.loads(p))
    return out


def _stream(client, body, headers=None):
    with client.stream("POST", "/v1/chat/completions", json=body, headers=headers or {}) as r:
        assert r.status_code == 200
        return _events(r.read().decode())


def test_fixture_contract_holds(client):
    ev = _stream(client, FIXTURE["request"])
    rules = FIXTURE["rules"]
    assert ev[-1] == "[DONE]"
    usage_chunk = ev[-2]
    assert usage_chunk["choices"] == rules["usage_chunk_choices"]
    u = usage_chunk["usage"]
    assert sorted(u) == sorted(rules["usage_fields"])
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["completion_tokens"] > 0
    for c in ev[:-2]:
        assert "usage" in c and c["usage"] is None
        assert c["id"] == usage_chunk["id"]


def test_usage_matches_non_stream(client):
    body = dict(FIXTURE["request"])
    ev = _stream(client, body)
    non = client.post(
        "/v1/chat/completions", json={**body, "stream": False, "stream_options": None}
    ).json()
    assert ev[-2]["usage"] == non["usage"]


def test_without_option_no_usage(client):
    body = {k: v for k, v in FIXTURE["request"].items() if k != "stream_options"}
    ev = _stream(client, body)
    assert all("usage" not in c for c in ev[:-1])
    ev2 = _stream(client, {**body, "stream_options": {"include_usage": False}})
    assert all("usage" not in c for c in ev2[:-1])


def test_tool_call_stream_also_gets_usage(client):
    body = {
        **FIXTURE["request"],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "required",
    }
    ev = _stream(client, body)
    assert ev[-3]["choices"][0]["finish_reason"] == "tool_calls"
    assert ev[-2]["choices"] == [] and ev[-2]["usage"]["total_tokens"] > 0


def test_mid_stream_drop_sends_no_usage(client):
    ev = _stream(client, FIXTURE["request"], headers={"X-Stub-Fault": "drop:2"})
    assert "[DONE]" not in ev
    assert all(c.get("choices") != [] for c in ev)


def test_recorded_transcript_is_reproducible(client):
    """include_usage_stream.sse is a recorded transcript (id/created normalised).

    llm-platform-k8s replays this exact file through its gateway, so if the
    stub's wire format changes this test flags that the shared copy is stale.
    """
    import re

    with client.stream("POST", "/v1/chat/completions", json=FIXTURE["request"]) as r:
        raw = r.read().decode()
    raw = re.sub(r'"id":"[^"]+"', '"id":"chatcmpl-fixture"', raw)
    raw = re.sub(r'"created":\d+', '"created":0', raw)
    assert raw == (Path(__file__).parent / "fixtures" / "include_usage_stream.sse").read_text()
