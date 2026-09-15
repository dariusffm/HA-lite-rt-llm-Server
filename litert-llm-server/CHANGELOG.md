# Changelog

## 0.2.1 — 2026-09-15

- New option `context_length` (default 8192, range 2048-32768), passed
  through to `litert_lm.Engine` as `max_num_tokens`. Fixes Home Assistant
  Assist prompts exceeding the previous hardcoded 4096-token library
  default.
- Streaming errors from the engine are now returned as clean Ollama/OpenAI
  error records instead of dropping the connection mid-response.

## 0.2.0 — 2026-09-15

- Client-side tool calling on both APIs: `tools` in requests, `tool_calls` in
  responses, `tool` role for results. Enables Home Assistant Assist (device
  control) and MCP-server tools (web search, weather, news).
- New option `tool_calling` (default `true`); `false` ignores tools.
- `litert-lm-api` 0.17.0.

## 0.1.3 — 2026-09-15

- `/api/tags` entries now carry `model` in addition to `name`, as real
  Ollama does. Home Assistant's Ollama integration reads `model` and
  crashed with `KeyError` when adding a conversation agent.

## 0.1.2 — 2026-09-15

- Fix `/api/pull` still failing after 0.1.1 when a dangling model symlink
  (left behind by 0.1.0) exists: `FilesystemCache.delete` now removes
  dangling symlinks before the pulled file is materialized.

## 0.1.1 — 2026-09-15

- Fix `/api/pull` failing with `FileNotFoundError` on Linux: the HuggingFace
  snapshot symlink was hardlinked instead of the blob it points to
  (CPython issue 37612). The source path is now resolved before linking.
- Repository metadata points to the public Github repo; MIT license added.

## 0.1.0 — 2026-05-17

- Initial release.
- OpenAI- and Ollama-compatible APIs.
- Gemma 4 E2B/E4B (public) and Gemma 3n E2B/E4B (gated) via HuggingFace catalog.
- Auto-download via `/api/pull`.
- Model-native chat templating via litert_lm Conversation API.
- 32k context window.
- amd64 + aarch64 builds.
