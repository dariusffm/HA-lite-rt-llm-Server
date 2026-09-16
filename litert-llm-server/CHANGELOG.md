# Changelog

## 0.4.3 — 2026-09-16

- Diagnostics: the engine now logs generation timing at DEBUG (start,
  first/every-50th chunk, producer finish/failure, `cancel_process`/
  `conversation.close` duration) so a stalled generation can be traced from
  the logs. A client abort mid-generation now logs a WARNING with elapsed
  time, chunk count, text chars produced and any tool-call fragment.
- New option `generation_timeout` (seconds, default 120, `0` disables): a
  generation that runs longer than the budget without finishing is
  cancelled, the same WARNING is logged, and the stream ends with a
  `RuntimeError` instead of hanging until the client's own timeout.
- `cancel_process()` on abort or timeout now runs off the event-loop thread,
  so a slow C-side cancel can no longer stall the consumer coroutine.

## 0.4.2 — 2026-09-16

- Conversation reuse now compares tool-call *names* only when matching HA's
  echoed assistant reply. A repaired tool call (0.4.1) no longer breaks the
  continuation, so the closing round after an action reuses the held
  conversation instead of prefilling again.

## 0.4.1 — 2026-09-16

- Tool calls that carry no entity `name` although the user named the entity
  (Gemma 4 E2B: `HassTurnOn{domain: [light], device_class: [switch]}` for
  "mach die Wohnzimmer-Fenster-Lampe an") are repaired: the add-on inserts the
  catalogue name from HA's Assist prompt and the entity's real domain, and
  drops the guessed `device_class`. Skipped when the call already targets a
  name, area or floor, or when the user text matches no or several entities.
  Option `tool_call_repair` (default `true`); log `tool call repaired: …`.

## 0.4.0 — 2026-09-16

- The engine keeps the last conversation alive and, when Home Assistant sends
  the same conversation back with one new turn (a tool result or a follow-up
  question), appends only that turn instead of prefilling the whole prompt
  again. On the test host a tool round drops from ~80 s to a few seconds, so
  multi-round Assist requests finish inside HA's 300 s pipeline timeout.
- New option `conversation_ttl` (seconds, default 300): how long an idle
  conversation is kept; `0` disables reuse. One KV cache of the last prompt
  stays in memory for that long.
- Reuse is skipped (and logged as `conversation reuse skipped: <reason>`)
  when the model, tools or sampler settings differ, when more than one turn
  is new, or when HA trimmed the history. Stage-1 compaction calls never
  touch the held conversation.

## 0.3.3 — 2026-09-16

- Stage-1 relevance routing no longer mangles multi-word area and entity
  names (e.g. "Bad Oben"); a compacted prompt could previously drop the
  entity the user actually asked about.
- A stage-1 request that times out now always closes the inner decode
  instead of leaking the producer thread and its conversation on the
  single-slot engine.
- Unexpected stage-1 engine errors are now logged at `warning` (previously
  `info`, indistinguishable from an expected fallback such as a timeout).
- The stage-1 cache key now includes the available domains/areas, so a
  changed entity set no longer reuses a stale routing answer for the same
  question text.
- The add-on option `log_level: critical` (already offered by the schema)
  is now accepted instead of raising a startup error.
- A blank line between the entity list and any trailing prompt text is no
  longer swallowed when the static context is re-rendered.

## 0.3.2 — 2026-09-16

- The note the add-on writes into a compacted Assist prompt now tells the
  model to call `GetLiveContext` with the entity's listed domain and that
  temperature, humidity, prices and other readings are domain `sensor`.
  Gemma 4 E2B otherwise asks for `climate` and Home Assistant finds nothing.

## 0.3.1 — 2026-09-16

- The `log_level` option now applies to the add-on's own loggers. Previously
  only uvicorn's access log honoured it and every INFO line from the app
  (`tool calling: …`, `prompt compaction: …`) was silently dropped.

## 0.3.0 — 2026-09-16

- New option `prompt_compaction` (`off` | `on` | `auto`, default `auto`).
  For Home Assistant Assist requests the add-on first asks the model which
  domains, areas or names the question is about (short JSON call), then
  runs the real turn with the system prompt reduced to those entities and
  `GetLiveContext` results rewritten to one line per entity. `auto`
  compacts only while `context_length` is below 16384. Requests without
  HA's Assist prompt are untouched; any parsing or stage-1 problem falls
  back to the original prompt and logs the reason.
- `GenerationParams.response_pattern`: engines enforce a regex on the reply via
  constrained decoding when no tools are offered.

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
