# openai-compat-inference-stub

Minimal **OpenAI-compatible** chat completions serving stub built with **FastAPI**.

> **Important:** This is an educational / open-source learning project only. It is
> **not** production software and does **not** represent any employer's (including
> Lowe's) production inference systems, gateways, or architectures.

## What it does

- `POST /v1/chat/completions` — OpenAI-ish request/response shape
- **Mock model** — deterministic reply derived from `messages` (no GPU, no network)
- **Latency + TTFT metrics** — `X-Latency-Ms` / `X-TTFT-Ms` headers, `latency_ms` on the JSON body
- **Prometheus text** — `GET /metrics` (`text/plain; version=0.0.4`) for scrapers; JSON aggregates at `GET /metrics.json`
- **Streaming** — `stream=true` returns OpenAI-compatible SSE (`data: {chunk}\n\n` … `data: [DONE]`)
- **Tool calls** — request `tools` / `tool_choice` → deterministic mock `tool_calls` with `finish_reason: tool_calls` (JSON + streamed argument deltas)
- **Fault injection (opt-in)** — deterministic 429 + `Retry-After`, 503, timeout, and mid-stream drop, plus synthetic `x-ratelimit-*` headers
- **Responses API (minimal, stateless)** — `POST /v1/responses` with JSON or typed SSE events (`response.created` … `response.completed`)
- `GET /health` — liveness

> **Honesty:** This is an OSS/learning stub only. It is not a production inference gateway and does not claim employer (or any vendor) production parity.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload --port 8000

# Chat completion
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "stub-model",
    "messages": [
      {"role": "system", "content": "You are a stub."},
      {"role": "user", "content": "Hello"}
    ]
  }' | python -m json.tool

# Streaming (SSE)
curl -N -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -D - \
  -d '{
    "model": "stub-model",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello stream"}]
  }'

# Health + metrics (Prometheus text + JSON)
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/metrics
curl -s http://127.0.0.1:8000/metrics.json | python -m json.tool
```

> **Honesty:** Hand-rolled Prometheus exposition for learning — not the official
> `prometheus_client` library, not a production inference gateway, and not employer
> (or vendor) production parity. JSON `/metrics.json` remains for humans; scrapers
> speak Prometheus text at `/metrics`.




## Tool calls (`tools` / `tool_choice`)

When the request includes OpenAI-style `tools`, the stub returns a deterministic
mock `tool_calls` message (`finish_reason: tool_calls`) instead of text. With
`stream=true`, SSE deltas use a stable `tool_calls[].index`, send `id` +
`function.name` on the first delta, then incremental `function.arguments`
pieces — the wire shape agent clients (LangChain / GPTMock-style) break on when
mocked incorrectly.

```bash
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "stub-model",
    "messages": [{"role": "user", "content": "find Paris"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "search",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }]
  }' | python -m json.tool
```

`tool_choice: "none"` forces a normal text completion even when `tools` is set.

> **Honesty:** Deterministic mock tool planner for offline agent/gateway CI —
> not a real model, not full OpenAI feature parity, not employer inference.

## Responses API (`POST /v1/responses`, minimal + stateless)

Newer clients (Codex CLI, recent SDKs, agent frameworks) speak the Responses API
rather than Chat Completions. The open requests to support it are some of the most
upvoted issues on local inference servers:
[llama.cpp#19138](https://github.com/ggml-org/llama.cpp/issues/19138),
[ollama#10309](https://github.com/ollama/ollama/issues/10309),
[ollama#9659](https://github.com/ollama/ollama/issues/9659),
[vllm#14721](https://github.com/vllm-project/vllm/issues/14721), and the
[vllm#57499 "stateless Responses API" RFC](https://github.com/vllm-project/vllm/issues/57499).
This stub adds a small, deterministic subset you can point such clients at in CI.

```bash
curl -s localhost:8000/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"stub-model","instructions":"be brief","input":"Hello"}'

# typed SSE events (no data: [DONE] in this API)
curl -sN localhost:8000/v1/responses -H 'Content-Type: application/json' \
  -d '{"input":"Hello","stream":true}'
```

| Supported | Notes |
|---|---|
| `input` as a string or list of items | message items (`content` string or `input_text`/`output_text` parts; `developer` treated as `system`), `function_call`, `function_call_output`; `reasoning` items are ignored |
| `instructions` | becomes the system prompt |
| `tools` with `type: function` | flat Responses tool shape; other tool types are ignored and listed in `X-Stub-Ignored-Tools` |
| `tool_choice` | `auto` / `none` / `required` / `{"type":"function","name":...}` |
| `max_output_tokens` | truncation gives `status: "incomplete"`, `incomplete_details.reason: "max_output_tokens"` |
| `stream: true` | `response.created` → `response.in_progress` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta`… → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → `response.completed` (or `response.incomplete`); function calls stream `response.function_call_arguments.delta` / `.done`. Each event carries `sequence_number`. |
| Faults | `X-Stub-Fault` / `model::fault=` apply as on chat (`429`, `503`, `timeout`, `drop:k` = stop after k delta events with no terminal event) |

**Stateless by design:** nothing is stored, every response reports `store: false`,
and `previous_response_id` returns a 400 (`code: previous_response_not_supported`).
Resend the whole conversation in `input`, including the `function_call` and `function_call_output`
items. IDs (`resp_…`, `msg_…`, `fc_…`) are a hash of the request, so the same
request gives the same ids and output. After a `function_call_output` as the last
input item, the stub answers in text (unless `tool_choice` forces a tool), so agent loops terminate.

> **Honesty:** A learning-sized subset of the Responses API (no built-in tools,
> no storage, no reasoning items, no structured `text.format`). It is not OpenAI parity.
> For a fuller mock, see [axium-lab/llm-mock](https://github.com/axium-lab/llm-mock).

## Fault injection (429 / 503 / timeout / mid-stream drop)

Retry, backoff, and fallback code is only trustworthy if you can make the upstream fail **on purpose, the same way every time**. The stub can inject OpenAI-shaped failures deterministically. They are **off by default**.

| Spec | Result |
|---|---|
| `429[:s]` | `429` + `{"error":{"code":"rate_limit_exceeded",…}}` + `Retry-After: s` (default 1) + `x-ratelimit-remaining-requests: 0` |
| `503[:s]` | `503` + `service_unavailable` (sends `Retry-After` only if `s` is given) |
| `timeout[:ms]` | holds the request `ms` (default `STUB_FAULT_TIMEOUT_MS`, i.e. 30000) so the client timeout fires, then answers normally |
| `drop[:k]` | `stream=true` only: sends `k` content chunks, then ends the stream **with no `finish_reason` and no `data: [DONE]`** (default k=2) |

**Triggers** (first match wins):
1. Request header `X-Stub-Fault: 429:2`. `X-Stub-Fault: none` disables env faults for that request.
2. Model suffix `"model": "stub-model::fault=503"`, for clients that can't set headers. The suffix is stripped from the echoed model id.
3. Env `STUB_FAULT=<spec>`, scheduled by a deterministic request counter: `STUB_FAULT_FIRST=N` (first N requests only, the retry-then-succeed shape) and/or `STUB_FAULT_EVERY=N` (requests N, 2N, …). With neither set, every request gets the fault.

Every chat response also carries **synthetic `x-ratelimit-*` headers** with OpenAI's names: `limit/remaining/reset` for `requests` and `tokens`, from a 60s window that is counted but never enforced. Limits come from `STUB_RATELIMIT_REQUESTS` (60) and `STUB_RATELIMIT_TOKENS` (100000). Injected faults are counted in `stub_faults_injected_total{kind=…}` on `/metrics` and in `faults_injected` on `/metrics.json`. A response with an injected fault includes `X-Stub-Fault: <kind>`.

```bash
# 429 with Retry-After + rate-limit headers
curl -si -X POST localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'X-Stub-Fault: 429:2' -d '{"messages":[{"role":"user","content":"hi"}]}'

# First 2 requests fail, then healthy (test your retry loop)
STUB_FAULT=429:1 STUB_FAULT_FIRST=2 uvicorn app.main:app --port 8000

# Stream that dies after 1 chunk (no [DONE])
curl -sN -X POST localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -H 'X-Stub-Fault: drop:1' -d '{"stream":true,"messages":[{"role":"user","content":"hi"}]}'

pytest tests/test_faults.py -q
```

> **Honesty:** synthetic failures and made-up header values. These are not OpenAI's real limits, and the stub is not a real rate limiter. It exists to exercise client retry and fallback code. The motivation is the stream of 2025–26 single-purpose mock servers built for exactly this: [LLMock](https://github.com/JulienRabault/LLMock), [roy](https://github.com/masci/roy), [faultkit](https://github.com/faultkit/faultkit), and [mocklimit](https://github.com/stano45/mocklimit).

## Configuration

See [`.env.example`](.env.example). All settings are optional; defaults work fully offline.

| Variable         | Default       | Meaning                                      |
|------------------|---------------|----------------------------------------------|
| `STUB_MODEL_ID`  | `stub-model`  | Default model id when request omits `model`  |
| `STUB_LATENCY_MS`| `0`           | Optional artificial delay for latency demos  |
| `STUB_HOST`      | `0.0.0.0`     | Documented for local run scripts             |
| `STUB_PORT`      | `8000`        | Documented for local run scripts             |
| `STUB_FAULT`     | _(unset)_     | Fault spec (see Fault injection); off by default |
| `STUB_FAULT_FIRST` / `STUB_FAULT_EVERY` | _(unset)_ | Deterministic schedule for `STUB_FAULT` |

## Tests

```bash
pytest -q
```

Runs fully offline via FastAPI `TestClient` / httpx (no live server required).

## CI

Workflow definition: [`ci/github-actions.yml`](ci/github-actions.yml).
To enable GitHub Actions, copy it to `.github/workflows/ci.yml` (requires a token
with the `workflow` scope to push that path).

## Project layout

```
app/
  main.py        # FastAPI routes
  responses_api.py # Minimal stateless /v1/responses mapping + typed SSE events
  schemas.py     # OpenAI-ish pydantic models
  mock_model.py  # Deterministic mock generator
  metrics.py     # In-process latency counters
  faults.py      # Deterministic fault injection + synthetic x-ratelimit-* headers
tests/           # pytest + TestClient
ci/              # GitHub Actions mirror
```

## License

MIT — see [LICENSE](LICENSE).
