# Changelog

## 0.6.0 — 2026-09-20

- Model names from HTTP requests are validated before they become
  filesystem paths. `DELETE /api/delete` passed the name through unchanged,
  so `../name` reached outside `/data/models`; the port is published on the
  LAN without authentication. Invalid names now answer HTTP 400.
- `HF_TOKEN` is persisted for s6 again. The filter matched a bare
  `HF_TOKEN` line that `printenv` never emits, so the token never reached
  uvicorn and gated model downloads failed with 401. The file is no longer
  world-readable.
- `default_model`, `max_tokens` and `temperature` from the add-on options
  now reach both adapters; they were read but never used, and the adapters
  applied their own literals instead. Values sent by a client still win.
  `model` may be omitted from a request and falls back to `default_model`.
- `preload_models` is pulled on startup, in a background task so the port
  binds immediately, with failures reported instead of silently drained.
- `max_tokens` now also bounds replies on the chat path. It is enforced
  while consuming tokens rather than when the conversation is created, so
  a reused conversation does not inherit an earlier request's limit.
- `/v1/completions` honours `stream: true` with SSE instead of answering
  with one late JSON block.
- A missing model no longer reports its absolute container path.

## 0.5.1 — 2026-09-20

- Clear the loaded-model state before closing/replacing an engine. If close
  or loading the replacement fails, the next request reloads the model
  instead of attempting to reuse an already closed engine.
- Scope prompt-compaction cache entries by model to prevent decisions from
  one model being reused for another. Cache hits now refresh recency so
  frequently used entries survive eviction.
- Include HassTurnOn in the Python settings defaults, matching the add-on's
  switching protection when starting the application directly.
- Add regression coverage for failed model switches, cache isolation and
  eviction, and agreement between Python and add-on switching defaults.

## 0.5.0 — 2026-09-17

- `HassTurnOn` is a switching tool by default. An untargeted `HassTurnOn`
  switched on *every* matching entity the same way `HassTurnOff` switched them
  off; only the off direction was guarded. Repair still runs first, so a call
  the user's text can pin to one entity is executed as before — only calls
  with no target at all are blocked.
- Tool specs are matched by the bare name as well, so repair keeps working if
  Home Assistant ever namespaces a call differently from its spec.
- Fix: `preload_models` had the same defect as `switching_tool_names` before
  0.4.9 — `bashio::config` prints a list one raw element per line and only the
  first line survived into the environment, so a configured model list broke
  the start. Both options now go through one `config_list` helper.

## 0.4.9 — 2026-09-17

- Fix: untargeted switching calls were never blocked on a real HA host.
  Home Assistant namespaces the tools it offers (`intent__HassTurnOff`,
  `homeassistant__GetLiveContext`), so the exact-name match against
  `switching_tool_names` never hit and `HassTurnOff{domain: [light]}` ran and
  turned off every light. Tool names are now compared without the namespace
  prefix, on both the call and the configured names.
- Fix: `LITERT_SWITCHING_TOOL_NAMES` was exported through `jq -c`, but
  `bashio::config` prints a list as one raw element per line, not as JSON —
  `jq` aborted with a parse error and the variable stayed empty (the service
  silently fell back to its built-in defaults). The init script now builds
  the JSON array from the lines and logs the result as `switching tools: …`.

## 0.4.8 — 2026-09-17

- Fix: the add-on crashed on start with `SettingsError: error parsing value
  for field "switching_tool_names"`. `bashio::config` prints a list as
  multi-line JSON, and the loop that persists the env vars for s6 reads only
  the first line of a value, so `LITERT_SWITCHING_TOOL_NAMES` was stored as a
  bare `[`. The init script now compacts the list with `jq -c` before
  exporting it. 0.4.6/0.4.7 never started on a host with s6 persistence.

## 0.4.7 — 2026-09-17

- Re-release of 0.4.6 with no code change; the add-on version had to be
  bumped for the Supervisor to offer the update. The FastAPI/OpenAPI
  `version` string, which had been left at 0.4.5 in 0.4.6, now matches again.

## 0.4.6 — 2026-09-17

- Switching tool calls without a target are blocked: a call to a tool from
  `switching_tool_names` (default `HassTurnOff`, `HassToggle`) that carries
  no `name`, `area` or `floor` and cannot be repaired from the user text is
  not executed. Unless the current or previous user message says "alle", the
  model receives `switching_block_reply` (default "Welches Gerät oder welchen
  Bereich meinst du genau?") instead. Mixed replies keep the allowed calls;
  log: `tool call blocked: <tool> without name/area/floor`.
- New options `switching_tool_names` and `switching_block_reply`.

## 0.4.5 — 2026-09-17

- Prompt compaction: a valid stage-1 reply that selects no entities (empty
  lists, only unknown domains, or no matching entity) now compacts to zero
  entities and is cached, instead of falling back to the full prompt. On the
  HA host the full prompt (176 entities) did not produce a first chunk
  within `generation_timeout`, so small talk and general questions ("Erzähl
  mir einen Witz", "Was ist 17 mal 23?") ended in "Unknown error". Real
  stage-1 failures (timeout, engine error, unparseable reply) still fall
  back to the full prompt.

## 0.4.4 — 2026-09-16

- Diagnostics: chunks from `send_message_async` that carry neither text nor
  a tool call (e.g. thinking content) are now logged at DEBUG (first three,
  then every 50th) so a decode that produces only such chunks no longer
  looks like silence. Found on a HA host where a tool round timed out after
  120 s with zero visible chunks.

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
