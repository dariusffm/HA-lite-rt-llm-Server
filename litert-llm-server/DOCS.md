# LiteRT LLM Server — Detailed Docs

## Endpoints

### OpenAI-compatible (`/v1/*`)

- `GET /v1/models`
- `POST /v1/chat/completions` (stream / non-stream, uses model-native chat template)
- `POST /v1/completions` (legacy raw prompt completion)

### Ollama-compatible (`/api/*`)

- `GET /api/tags`
- `POST /api/chat` (stream / non-stream NDJSON, uses model-native chat template)
- `POST /api/generate` (stream / non-stream NDJSON, raw prompt)
- `POST /api/pull` (stream progress NDJSON)
- `POST /api/show`
- `DELETE /api/delete`

### Health

- `GET /healthz` — liveness (always 200)
- `GET /readyz` — readiness, reports whether at least one model is cached

## Using with Home Assistant

Add an "OpenAI Conversation" integration:

- **Base URL:** `http://<add-on-hostname>:8080/v1`
- **API key:** anything (no auth in MVP)
- **Model:** `gemma-4-e2b`

## Using with Node-RED

Use any Ollama node and point it to `http://<add-on-hostname>:8080`.

## Pulling a Model

The add-on does not bundle models. Use either:

- The add-on UI (web ingress, planned) to trigger a pull
- Or curl directly:
  ```bash
  curl -X POST http://<add-on-hostname>:8080/api/pull \
    -H 'content-type: application/json' \
    -d '{"name":"gemma-4-e2b","stream":false}'
  ```

For gated models, set the `hf_token` option first.

## Limits (MVP)

- **Context window: 32k tokens** (prompt + completion combined). Long
  multi-turn conversations may exhaust it; no automatic truncation in MVP.
- Single-slot engine: switching models mid-flight triggers a reload.
- No request queue: concurrent requests serialize.
- No authentication: rely on HA's internal network.
- Embeddings, function-calling, and multi-modal are not yet supported.
