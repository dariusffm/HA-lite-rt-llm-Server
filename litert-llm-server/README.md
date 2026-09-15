# LiteRT LLM Server

Local LLM inference via Google LiteRT-LM. Exposes OpenAI- and
Ollama-compatible HTTP APIs on port 8080, consumable by Home Assistant's
"OpenAI Conversation" integration and any Ollama-compatible client
(Node-RED nodes, Open WebUI, ...).

## Supported Models (MVP)

All models use the LiteRT-LM `.litertlm` format. Context window: up to
32k tokens (prompt + completion combined).

| Name | Repository | Gated | File Size | Tool calling |
|---|---|---|---|---|
| `gemma-4-e2b` (default) | `litert-community/gemma-4-E2B-it-litert-lm` | no | ~2.6 GB | yes |
| `gemma-4-e4b` | `litert-community/gemma-4-E4B-it-litert-lm` | no | ~3.7 GB | yes |
| `gemma-3n-e2b` | `google/gemma-3n-E2B-it-litert-lm` | yes (HF token) | ~3.0 GB | not documented |
| `gemma-3n-e4b` | `google/gemma-3n-E4B-it-litert-lm` | yes (HF token) | ~4.2 GB | not documented |

Sizes are the actual `.litertlm` files (verified on a live install for
`gemma-4-e2b`: 2,588,147,712 bytes). Models are downloaded on demand via the
Ollama-compatible `/api/pull` endpoint. See `DOCS.md` for usage and for the
list of further `.litertlm` models the runtime can load (Qwen3, Phi-4-mini,
Gemma 4 12B, FunctionGemma, …) that are not yet in the catalog.

## Prerequisites — HuggingFace Access (only for gated Gemma 3n models)

The `litert-community/*` Gemma 4 models are public — no token needed.
The `google/*` Gemma 3n models are **gated**. To use them:

1. Sign in at <https://huggingface.co/>.
2. Open the model page (e.g.
   <https://huggingface.co/google/gemma-3n-E2B-it-litert-lm>) and accept
   the Gemma license.
3. Create a **read-scope access token** at
   <https://huggingface.co/settings/tokens>.
4. Paste the token into the `hf_token` add-on option (stored as a
   password-type field, masked in HA's UI and never logged).

Without a valid token, `/api/pull` requests for gated models will fail
with HTTP 401.

## Configuration

| Option | Default | Description |
|---|---|---|
| `log_level` | `info` | trace, debug, info, notice, warning, error, fatal |
| `default_model` | `gemma-4-e2b` | Model used when a request omits `model` |
| `max_tokens` | 1024 | Default upper bound; can be raised up to 32768 (the LiteRT-LM context window) |
| `temperature` | 0.7 | Default sampling temperature |
| `preload_models` | `[]` | Model names to pull on startup |
| `hf_token` | `""` | HuggingFace read token; required only for gated `google/*` models |
