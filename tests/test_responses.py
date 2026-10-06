"""Minimal stateless /v1/responses: JSON shape, typed SSE events, tools, faults."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.pop("STUB_LATENCY_MS", None)
os.environ["STUB_MODEL_ID"] = "stub-model"

from app.faults import faults, rate_limits  # noqa: E402
from app.main import app  # noqa: E402
from app.metrics import metrics  # noqa: E402

FAULT_ENV = ("STUB_FAULT", "STUB_FAULT_FIRST", "STUB_FAULT_EVERY", "STUB_FAULT_TIMEOUT_MS",
             "STUB_RATELIMIT_REQUESTS", "STUB_RATELIMIT_TOKENS")

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in FAULT_ENV:
        monkeypatch.delenv(k, raising=False)
    faults.reset()
    rate_limits.reset()
    metrics.request_count = 0
    metrics.error_count = 0
    metrics.stream_request_count = 0
    yield
    faults.reset()
    rate_limits.reset()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def parse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: "), block
        assert lines[1].startswith("data: "), block
        name = lines[0][len("event: "):]
        data = json.loads(lines[1][len("data: "):])
        assert data["type"] == name
        events.append((name, data))
    return events


def test_string_input_returns_response_object(client: TestClient):
    r = client.post("/v1/responses", json={"input": "hello responses", "instructions": "be brief"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["id"].startswith("resp_")
    assert body["store"] is False
    assert body["instructions"] == "be brief"
    (item,) = body["output"]
    assert item["type"] == "message" and item["role"] == "assistant"
    assert item["id"].startswith("msg_")
    (part,) = item["content"]
    assert part["type"] == "output_text" and part["annotations"] == []
    assert "echo=hello responses" in part["text"]
    assert "system_hint=be brief" in part["text"]
    u = body["usage"]
    assert u["total_tokens"] == u["input_tokens"] + u["output_tokens"]
    assert r.headers["x-stub-model"] == "stub-model"


def test_deterministic_ids_and_output(client: TestClient):
    payload = {"input": [{"role": "user", "content": "same request"}]}
    a = client.post("/v1/responses", json=payload).json()
    b = client.post("/v1/responses", json=payload).json()
    assert a["id"] == b["id"]
    assert a["output"] == b["output"]
    c = client.post("/v1/responses", json={"input": "different request"}).json()
    assert c["id"] != a["id"]


def test_message_items_with_content_parts_and_developer_role(client: TestClient):
    payload = {
        "input": [
            {"role": "developer", "content": "dev hint"},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "part one "},
                         {"type": "input_text", "text": "part two"}]},
        ]
    }
    body = client.post("/v1/responses", json=payload).json()
    text = body["output"][0]["content"][0]["text"]
    assert "echo=part one part two" in text
    assert "system_hint=dev hint" in text


def test_string_and_equivalent_message_list_give_same_output(client: TestClient):
    a = client.post("/v1/responses", json={"input": "hi"}).json()
    b = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hi"}]}).json()
    assert a["output"][0]["content"] == b["output"][0]["content"]


def test_function_tool_call_and_ignored_tool_types(client: TestClient):
    r = client.post(
        "/v1/responses",
        json={"input": "weather in Paris", "tools": [WEATHER_TOOL, {"type": "web_search"}]},
    )
    assert r.status_code == 200
    assert r.headers["x-stub-ignored-tools"] == "web_search"
    (item,) = r.json()["output"]
    assert item["type"] == "function_call"
    assert item["name"] == "get_weather"
    assert item["id"].startswith("fc_")
    assert item["call_id"].startswith("call_")
    assert isinstance(json.loads(item["arguments"]), dict)


def test_tool_choice_none_and_named(client: TestClient):
    none = client.post(
        "/v1/responses",
        json={"input": "weather", "tools": [WEATHER_TOOL], "tool_choice": "none"},
    ).json()
    assert none["output"][0]["type"] == "message"
    other = {**WEATHER_TOOL, "name": "get_time"}
    named = client.post(
        "/v1/responses",
        json={"input": "time", "tools": [WEATHER_TOOL, other],
              "tool_choice": {"type": "function", "name": "get_time"}},
    ).json()
    assert named["output"][0]["name"] == "get_time"


def test_tool_round_trip_terminates_with_text(client: TestClient):
    first = client.post(
        "/v1/responses", json={"input": "weather in Oslo", "tools": [WEATHER_TOOL]}
    ).json()
    call = first["output"][0]
    second = client.post(
        "/v1/responses",
        json={
            "input": [
                {"role": "user", "content": "weather in Oslo"},
                {"type": "function_call", "call_id": call["call_id"],
                 "name": call["name"], "arguments": call["arguments"]},
                {"type": "function_call_output", "call_id": call["call_id"],
                 "output": json.dumps({"temp_c": 4})},
            ],
            "tools": [WEATHER_TOOL],
        },
    )
    assert second.status_code == 200
    (item,) = second.json()["output"]
    assert item["type"] == "message"


def test_max_output_tokens_gives_incomplete(client: TestClient):
    body = client.post(
        "/v1/responses",
        json={"input": "a fairly long prompt with many words in it", "max_output_tokens": 2},
    ).json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_previous_response_id_is_rejected(client: TestClient):
    r = client.post("/v1/responses", json={"input": "hi", "previous_response_id": "resp_x"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "previous_response_not_supported"
    assert err["param"] == "previous_response_id"


@pytest.mark.parametrize(
    "bad_input",
    [
        [],
        [{"type": "computer_call_output", "call_id": "x"}],
        [{"role": "wizard", "content": "hi"}],
        [{"role": "user", "content": [{"type": "input_audio", "data": "..."}]}],
    ],
)
def test_invalid_input_items_are_400(client: TestClient, bad_input):
    r = client.post("/v1/responses", json={"input": bad_input})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_stream_text_event_sequence(client: TestClient):
    r = client.post("/v1/responses", json={"input": "stream me please", "stream": True})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" not in r.text
    events = parse_events(r.text)
    names = [n for n, _ in events]
    assert names[:4] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
    ]
    assert names[-4:] == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    deltas = [d for n, d in events if n == "response.output_text.delta"]
    assert len(deltas) >= 2
    assert [d["sequence_number"] for _, d in events] == list(range(len(events)))
    done_text = next(d["text"] for n, d in events if n == "response.output_text.done")
    assert "".join(d["delta"] for d in deltas) == done_text
    assert events[0][1]["response"]["status"] == "in_progress"
    final = events[-1][1]["response"]
    assert final["status"] == "completed"
    assert final["output"][0]["content"][0]["text"] == done_text
    # stream final == non-stream body (except created_at)
    plain = client.post("/v1/responses", json={"input": "stream me please"}).json()
    assert final["id"] == plain["id"] and final["output"] == plain["output"]


def test_stream_function_call_events(client: TestClient):
    r = client.post(
        "/v1/responses",
        json={"input": "weather in Rome", "tools": [WEATHER_TOOL], "stream": True},
    )
    events = parse_events(r.text)
    names = [n for n, _ in events]
    assert "response.output_text.delta" not in names
    assert "response.function_call_arguments.delta" in names
    args = "".join(d["delta"] for n, d in events if n == "response.function_call_arguments.delta")
    done = next(d for n, d in events if n == "response.function_call_arguments.done")
    assert args == done["arguments"]
    added = next(d for n, d in events if n == "response.output_item.added")
    assert added["item"]["type"] == "function_call" and added["item"]["arguments"] == ""
    assert names[-1] == "response.completed"
    assert events[-1][1]["response"]["output"][0]["arguments"] == args


def test_stream_incomplete_terminal_event(client: TestClient):
    r = client.post(
        "/v1/responses",
        json={"input": "a long enough prompt to truncate", "max_output_tokens": 2, "stream": True},
    )
    events = parse_events(r.text)
    assert events[-1][0] == "response.incomplete"
    assert events[-1][1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


def test_fault_429_and_503(client: TestClient):
    r = client.post("/v1/responses", json={"input": "hi"}, headers={"X-Stub-Fault": "429:2"})
    assert r.status_code == 429
    assert r.headers["retry-after"] == "2"
    r = client.post("/v1/responses", json={"model": "stub-model::fault=503", "input": "hi"})
    assert r.status_code == 503


def test_fault_drop_stops_without_terminal_event(client: TestClient):
    r = client.post(
        "/v1/responses",
        json={"input": "stream then drop please", "stream": True},
        headers={"X-Stub-Fault": "drop:1"},
    )
    events = parse_events(r.text)
    names = [n for n, _ in events]
    assert names.count("response.output_text.delta") == 1
    assert "response.completed" not in names
    assert "response.output_text.done" not in names


def test_invalid_fault_spec_is_400(client: TestClient):
    r = client.post("/v1/responses", json={"input": "hi"}, headers={"X-Stub-Fault": "bogus"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_fault_spec"


def test_metrics_count_responses_requests(client: TestClient):
    client.post("/v1/responses", json={"input": "one"})
    client.post("/v1/responses", json={"input": "two", "stream": True})
    snap = client.get("/metrics.json").json()
    assert snap["request_count"] == 2
    assert snap["stream_request_count"] == 1
