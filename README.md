# openai-compat-inference-stub

Minimal **OpenAI-compatible** chat completions serving stub built with **FastAPI**.

> **Important:** This is an educational / open-source learning project only. It is
> **not** production software and does **not** represent any employer's (including
> Lowe's) production inference systems, gateways, or architectures.

## What it does

- `POST /v1/chat/completions` — OpenAI-ish request/response shape
- **Mock model** — deterministic reply derived from `messages` (no GPU, no network)
- **Latency metrics** — `X-Latency-Ms` response header, `latency_ms` on the JSON body, and `GET /metrics`
- `GET /health` — liveness

Streaming (`stream=true`) is intentionally **not** supported.

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

# Health + metrics
curl -s http://127.0.0.1:8000/health | python -m json.tool
curl -s http://127.0.0.1:8000/metrics | python -m json.tool
```

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
