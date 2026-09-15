# LiteRT LLM Server — Detailed Docs

## Endpoints

### OpenAI-compatible (`/v1/*`)

- `GET /v1/models`
- `POST /v1/chat/completions` (stream / non-stream, uses model-native chat template)
- `POST /v1/completions` (legacy raw prompt completion)
- `/v1/chat/completions` accepts `tools` / `tool_choice` and returns
  `tool_calls` with `finish_reason: "tool_calls"`.

### Ollama-compatible (`/api/*`)

- `GET /api/tags`
- `POST /api/chat` (stream / non-stream NDJSON, uses model-native chat template)
- `POST /api/generate` (stream / non-stream NDJSON, raw prompt)
- `POST /api/pull` (stream progress NDJSON)
- `POST /api/show`
- `DELETE /api/delete`
- `/api/chat` accepts `tools` and returns `message.tool_calls` (client-side
  tool calling, Ollama wire format).

### Health

- `GET /healthz` — liveness (always 200)
- `GET /readyz` — readiness, reports whether at least one model is cached

## Using with Home Assistant

Use the core **Ollama** integration (Settings → Devices & Services → Add
Integration → Ollama). URL: `http://<addon-hostname>:8080` — the hostname is
shown on the add-on's Info page (e.g. `83680c0c-litert-llm-server`). Then add
a **Conversation agent** and pick the model (e.g. `gemma-4-e2b`).

*Home Assistant's "OpenAI Conversation" integration has no base-URL option and
cannot be pointed at this add-on.*

### Tool calling (control your home, web search)

Tool calling is on by default (`tool_calling: true`). The add-on never
executes tools itself; Home Assistant offers them and runs them:

- **Assist** — tick "Assist" in the conversation agent to let the model see
  and control exposed entities ("turn on the hallway light").
- **Web search / news / weather** — install a web-search MCP server (any
  server speaking Model Context Protocol, e.g. a DuckDuckGo or Brave Search
  MCP server, run as a container on your network), add it via Settings →
  Devices & Services → **Model Context Protocol**, then enable that MCP
  server's tools in the conversation agent's options. The model can then call
  the search tool; Home Assistant performs the request and feeds the result
  back.

Set `tool_calling: false` to ignore all tools: replies are plain text as in
0.1.x, even if the agent has Assist or MCP tools enabled.

Small models (Gemma 4 E2B) sometimes emit malformed tool arguments; Home
Assistant repairs common cases. Keep the number of exposed entities small to
save context.

Home Assistant's agent option *Größe des Kontextfensters* (`num_ctx`) must
not exceed the add-on's `context_length`; the add-on ignores `num_ctx`. With
many exposed entities raise `context_length` (more RAM).

## Using with Node-RED

Use any Ollama node and point it to `http://<add-on-hostname>:8080`.

Tool calling requires a client that executes tools; Node-RED's Ollama nodes do
not, so tool results are not fed back there yet.

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

## Model Landscape (verified 2026-09-15, litert-lm-api 0.17.0)

The add-on can only `/api/pull` models listed in its catalog
(`model_registry/huggingface.py`). The runtime itself loads any `.litertlm`
file; the tables below show what exists on HuggingFace today so the catalog
can grow. Sizes come from the HF file listings or Google's LiteRT-LM overview
table; "Tool calling" means Google documents native function calling.

### In the catalog (pullable now)

| Name | Repo | File | Size | Gated | Context | Tool calling |
|---|---|---|---|---|---|---|
| `gemma-4-e2b` | `litert-community/gemma-4-E2B-it-litert-lm` | `gemma-4-E2B-it.litertlm` | 2.6 GB | no | 32k | yes |
| `gemma-4-e4b` | `litert-community/gemma-4-E4B-it-litert-lm` | `gemma-4-E4B-it.litertlm` | 3.7 GB | no | 32k | yes |
| `gemma-3n-e2b` | `google/gemma-3n-E2B-it-litert-lm` | `gemma-3n-E2B-it-int4.litertlm` | 3.0 GB | yes | 32k | not documented |
| `gemma-3n-e4b` | `google/gemma-3n-E4B-it-litert-lm` | `gemma-3n-E4B-it-int4.litertlm` | 4.2 GB | yes | 32k | not documented |

### Available as `.litertlm`, not yet in the catalog (candidates)

| Candidate | Repo | File | Size | Gated | Notes |
|---|---|---|---|---|---|
| Gemma 4 12B | `litert-community/gemma-4-12B-it-litert-lm` | `gemma-4-12B-it.litertlm` | 6.9 GB | no | 128k context, multimodal, tool calling; needs LiteRT-LM ≥ 0.17 (we have it). Too large for most HA hosts. |
| Gemma 3 1B | `litert-community/Gemma3-1B-IT` | `gemma3-1b-it-int4.litertlm` (+ device variants) | ~1.0 GB | yes | Small and fast; no tool calling. |
| Gemma 3 270M | `litert-community/gemma-3-270m-it` | `gemma3-270m-it-q8.litertlm` | 0.3 GB | yes | Tiny; classification/short replies only. |
| FunctionGemma 270M (mobile actions) | `litert-community/functiongemma-270m-ft-mobile-actions` | `mobile_actions_q8_ekv1024.litertlm` | ~0.3 GB | yes | Google's dedicated on-device function-calling model; 1k KV cache. Good for pure tool routing, not for chat. |
| Qwen3 0.6B | `litert-community/Qwen3-0.6B` | `Qwen3-0.6B.litertlm` (+ int4 variants in `Qwen3-0.6B-int4`) | 0.6 GB | no | Apache-2.0; tool calling not documented for the conversion. |
| Qwen3 4B | `litert-community/Qwen3-4B` | `qwen3_4b_mixed_int4.litertlm` | n/a | no | Apache-2.0. |
| Qwen2.5 1.5B | `litert-community/Qwen2.5-1.5B-Instruct` | `Qwen2.5-1.5B-Instruct_multi-prefill-seq_q8_ekv4096.litertlm` | 1.6 GB | no | 4k KV cache. |
| Phi-4-mini 3.8B | `litert-community/Phi-4-mini-instruct` | `Phi-4-mini-instruct_multi-prefill-seq_q8_ekv4096.litertlm` | 3.9 GB | no | MIT; 4k KV cache. |
| Llama 3.2 1B / 3B | `litert-community/Llama-3.2-{1B,3B}-Instruct` | `..._q8_ekv<N>.litertlm` (pattern) | n/a | yes (Meta license, repo not browsable anonymously) | Filenames unverified. |
| DeepSeek-R1-Distill-Qwen 1.5B | `litert-community/DeepSeek-R1-Distill-Qwen-1.5B` | `DeepSeek-R1-Distill-Qwen-1.5B_multi-prefill-seq_q8_ekv4096.litertlm` | n/a | no | Reasoning model; no tool calling. |
| SmolLM2 135M / 1.7B | `litert-community/SmolLM2-{135M,1.7B}-Instruct` | `SmolLM2_135M_Instruct.litertlm`, `SmolLM2-1_7B-Instruct_dynamic_wi8_afp32.litertlm` | 0.1 / 1.7 GB | no | Apache-2.0. |

Adding a candidate = one `CatalogEntry` (name, repo, filename, quantization,
gated) in `model_registry/huggingface.py` plus a README row. Files with
`ekv<N>` in the name have a fixed KV cache of N tokens — that caps the
usable context regardless of the 32k default.

### Not available as `.litertlm` (as of 2026-09-15)

- Gemma 3 4B (`litert-community/Gemma3-4B-IT`) — only MediaPipe `.task` files.
- Qwen2.5 0.5B, TinyLlama 1.1B — only `.tflite` / `.task`.
- SmolLM3 — no LiteRT-LM conversion found.
- EmbeddingGemma 300M (`litert-community/embeddinggemma-300m`) — `.tflite`
  only; a community `.litertlm` exists (`kontextdev/embeddinggemma-300m-litertlm`)
  but is unofficial. Embeddings are out of scope for this add-on anyway.

### Memory

LiteRT-LM publishes no minimum-RAM figures, only measured peak CPU memory
(≈0.6–3.5 GB depending on model and device). Rule of thumb: free RAM ≥ file
size + 1 GB. On the test host, `gemma-4-e2b` idles at ~2 % RAM and runs
with ~27 % CPU during a reply.

### Function calling

Google names two options: **Gemma 4** (E2B/E4B/12B) for agentic chat, and
**FunctionGemma 270M** for dedicated tool routing. The catalog default
`gemma-4-e2b` is therefore the right choice for the tool-calling feature.

## Limits (MVP)

- **Context window: `context_length` option, default 8192 tokens (model
  supports up to 32k); prompt + completion combined**. Long multi-turn
  conversations may exhaust it; no automatic truncation in MVP.
- Single-slot engine: switching models mid-flight triggers a reload.
- No request queue: concurrent requests serialize.
- No authentication: rely on HA's internal network.
- Embeddings and multi-modal are not yet supported. Function calling: see
  *Tool calling* above.
