# End-to-End Test — Tool calling (litert-llm-server 0.2.x)

**Date:** 2026-09-15
**HA instance:** HAOS with Supervisor, aarch64 host, add-on hostname `83680c0c-litert-llm-server`
**Versions tested:** 0.2.0 (first rollout), 0.2.1 (context window fix)
**Client:** Home Assistant core *Ollama* integration, conversation agent on `gemma-4-e2b`, **Assist enabled**

## Result

- Add-on updated to 0.2.0 / 0.2.1 via the Supervisor: **yes** (~2 min rebuild each)
- `/api/chat` returned `tool_calls` for the weather prompt (single tool, from the Mac): **yes**
- Tool-result round trip produced a sensible answer: **yes** — "The weather in Frankfurt right now is 21°C and sunny."
- HA Assist read access via the agent: **yes** — "Welche Lichter sind gerade eingeschaltet?" → model called `GetLiveContext`, HA returned the exposed light states, model answered with the two lights that are on (one by Zigbee id, one by name).
- HA Assist device control ("Schalte … ein"): **not run yet** — needs the user to name a device that is safe to toggle.
- MCP web search: **not run** — no web-search MCP server deployed yet (user decision, see TODO.md).

## Bugs found only in the real environment

| Version | Bug | Root cause / fix |
|---|---|---|
| 0.2.0 | Assist request failed: `RuntimeError: INVALID_ARGUMENT: Input token ids are too long … 6089 >= 4096`; HA showed "Unexpected error during intent recognition" (`httpx.RemoteProtocolError: incomplete chunked read`) | `LiteRTEngine` never set `max_num_tokens`, so the library default (4096) applied although the model supports 32k; the exception also cut the chunked response. 0.2.1 adds the `context_length` option (default 8192) and returns Ollama/OpenAI error records instead of dropping the stream. |

## Observations

- **Latency.** With Assist on, HA's prompt (system prompt + all tool schemas for the exposed entities) is ~6k tokens. On this host the full turn took roughly 3–4 minutes wall clock: prefill of the first request, the tool call, a second prefill including the `GetLiveContext` result, then the answer. Plain chat without tools answers in seconds. Reducing exposed entities is the main lever; prefill throughput on CPU is the bottleneck.
- The add-on's FastAPI `version` string still reports 0.2.0 on the 0.2.1 build (cosmetic, fixed in the simplify pass).
- Many exposed lights report `unavailable`; the model listed only the ones with `state: 'on'`, which is the correct reading of the tool result.
