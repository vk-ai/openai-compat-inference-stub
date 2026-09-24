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

## Configuration

See [`.env.example`](.env.example). All settings are optional; defaults work fully offline.

| Variable         | Default       | Meaning                                      |
|------------------|---------------|----------------------------------------------|
| `STUB_MODEL_ID`  | `stub-model`  | Default model id when request omits `model`  |
| `STUB_LATENCY_MS`| `0`           | Optional artificial delay for latency demos  |
| `STUB_HOST`      | `0.0.0.0`     | Documented for local run scripts             |
| `STUB_PORT`      | `8000`        | Documented for local run scripts             |

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
  schemas.py     # OpenAI-ish pydantic models
  mock_model.py  # Deterministic mock generator
  metrics.py     # In-process latency counters
tests/           # pytest + TestClient
ci/              # GitHub Actions mirror
```

## License

MIT — see [LICENSE](LICENSE).
