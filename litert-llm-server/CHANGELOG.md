# Changelog

## 0.2.4 — 2026-09-16

- Tool calls are now generated with LiteRT-LM constrained decoding whenever
  the client passes tools. Gemma 4 E2B otherwise emits arguments the
  library's grammar rejects (e.g. `{domain:light}` without quotes), which
  surfaced as `Failed to parse tool calls from code block` and an
  "Unexpected error during intent recognition" in Home Assistant. Plain-text
  replies and speed are unaffected.

## 0.2.3 — 2026-09-16

- Chat requests whose prompt exceeds `context_length` no longer fail with
  `Input token ids are too long`: the engine drops the oldest history round
  (user turn, tool calls and tool results) and retries until the prompt
  fits, logging a warning with the number of dropped turns. Only when the
  system prompt plus the latest message alone do not fit is the error
  returned to the client as before.
- A producer thread blocked on a full token queue now ends within about
  half a second after the client disconnects instead of hanging forever
  and keeping the conversation in memory.
- Failures while closing or cancelling a conversation are logged at debug
  level instead of being silently swallowed.

## 0.2.2 — 2026-09-15

- Internal cleanup after the tool-calling phase: shared tool-argument parsing
  and tool-call id minting in `domain/`, per-adapter error helpers, split
  completion/chat finish types. No functional change.
- FastAPI/OpenAPI version string now matches the add-on version.

## 0.2.1 — 2026-09-15

- New option `context_length` (default 8192, range 2048-32768), passed
  through to `litert_lm.Engine` as `max_num_tokens`. Fixes Home Assistant
  Assist prompts exceeding the previous hardcoded 4096-token library
  default.
- Engine errors during `/api/chat`, `/api/generate` and `/v1/chat/completions`
  streaming are now returned as a clean Ollama/OpenAI error record instead of
  dropping the connection mid-response. Non-streaming engine errors on those
  three endpoints, plus `/v1/completions` (which has no streaming branch),
  now return HTTP 500 with a JSON error body instead of an unhandled
  traceback response.

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
