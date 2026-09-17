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
Integration → Ollama). URL: `http://<add-on-hostname>:8080` — the hostname is
shown on the add-on's Info page (e.g. `83680c0c-litert-llm-server`). Then add
a **Conversation agent** and pick the model (e.g. `gemma-4-e2b`).

*Home Assistant's "OpenAI Conversation" integration has no base-URL option and
cannot be pointed at this add-on.*

### Recommended agent settings

Open the agent's options (gear icon on the Ollama integration page). These
values were tested with HA 2026.9 and `gemma-4-e2b` on a CPU-only host:

| Option | Value | Why |
|---|---|---|
| Modell | `gemma-4-e2b` | Native tool calling, 32k context, 2.6 GB. |
| Home Assistant steuern → Assist | on | Lets the model see and control exposed entities. |
| Größe des Kontextfensters | same as the add-on's `context_length` (default 8192) | The add-on ignores `num_ctx`; a larger value only hides the real limit. |
| Max. Nachrichten im Verlauf | 6 | HA resends the whole history each turn; more history means more prefill per turn. |
| Anweisungen | text below | Gemma 4 E2B needs explicit rules for filtered tool calls and short replies. |

Instructions to paste (English works best with Gemma):

```
You are a voice assistant for Home Assistant.
Answer questions about the world truthfully.
Respond simply and to the point in plain text.
When you need entity states, always call GetLiveContext with a domain, name or area filter (e.g. domain: light). Never call it without a filter.
After a successful action reply with one short sentence and call no further tools.
```

The last line matters: Home Assistant aborts an Assist run after 300 s
("Timeout running pipeline", not configurable in the UI). Every tool round
costs a full prefill of system prompt, tool definitions and history (about
80 s on the test host), so a third round hits the limit. Actions already
executed stay executed; only the closing sentence is lost. Since 0.4.0 follow-up rounds reuse the held conversation (see `conversation_ttl`), so the instruction is a safety net rather than the only defence.

Entities are found by **area** and **domain** filters only if they are
assigned in HA (Settings → Areas, and Settings → Voice assistants → Expose).
A temperature sensor without an area is invisible to "How warm is it in the
bathroom?" whatever the model asks for.

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

Home Assistant resends the whole conversation history with every message,
and Assist tool results (e.g. a `GetLiveContext` device list) are large. When
the prompt exceeds `context_length`, the add-on drops the oldest history
rounds until it fits and logs `prompt exceeds context window; dropped N
oldest history turn(s)`. The model then answers without that older context.
If this happens often, lower the agent option *Max. Nachrichten im Verlauf*
(e.g. 6) or expose fewer entities to Assist. Raise `context_length` **only
if the host has the RAM for it** — on the test host (HAOS, aarch64) 16384
made the process die right after model load, presumably OOM-killed, and the
add-on stayed stopped. Enable the add-on **Watchdog** so a killed process is
restarted automatically.

Set `tool_calling: false` to ignore all tools: replies are plain text as in
0.1.x, even if the agent has Assist or MCP tools enabled.

Tool calls are generated with constrained decoding, so the model cannot emit
malformed arguments (0.2.4). Keep the number of exposed entities small to save
context.

Home Assistant's `GetLiveContext` tool accepts optional `domain`, `name` and
`area` filters (HA 2026.9+, AND-combined). Gemma 4 E2B only uses them when
told to; see the instructions under *Recommended agent settings*. With
`prompt_compaction` active the add-on adds this hint itself, including the
rule that readings (temperature, humidity, prices) are domain `sensor`.

With many exposed entities either keep `prompt_compaction` on or raise
`context_length` (more RAM).

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
- `prompt_compaction` (`auto`): Home Assistant's Assist prompt lists every
  exposed entity and `GetLiveContext` results can be large. With compaction
  on, the add-on first asks the model (a short extra call, ~10–20 s on CPU
  hosts) which domains, areas or names the question concerns, then runs the
  turn with only those entities in the system prompt and Live Context results
  rewritten to one line per entity. `auto` enables this while `context_length`
  is below 16384; set `on` for slow hosts with large windows, `off` to always
  send the full prompt. Only requests carrying HA's Assist prompt are
  affected. If stage 1 fails, times out, returns nothing usable or no entity
  matches, the full prompt is used and the log says why:
  `prompt compaction skipped: …`. Successful runs log
  `prompt compaction: entities 176→14, tool turns compacted 1, stage-1 12.3s (cache miss)`.
- `conversation_ttl` (`300`): Home Assistant resends the whole conversation on
  every tool round. The add-on keeps the last conversation's KV cache alive
  and, when the next request is the same conversation plus one new turn,
  appends only that turn. Follow-up rounds then cost seconds instead of a
  full prefill (~80 s on the test host), which keeps multi-round requests
  inside HA's 300 s pipeline timeout. The cache is dropped after
  `conversation_ttl` idle seconds, on client abort, on errors and on model
  switch; `0` disables reuse. Log lines: `conversation reuse: appended 1 turn
  (kept 7475 tokens, idle 12.3s)` and `conversation reuse skipped: <reason>`.
  With `prompt_compaction` active, follow-up *questions* usually change the
  compacted entity list and run fresh; tool rounds reuse.
- `tool_call_repair` (`true`): if the model calls a tool that takes an entity
  `name` (HassTurnOn, HassTurnOff, …) without one, and the last user message
  contains exactly one entity name or alias from the Assist prompt, the add-on
  fills in that name and the entity's domain and drops a guessed
  `device_class`. Calls that already name a target, an area or a floor are
  left alone, as are ambiguous user texts. Log: `tool call repaired: HassTurnOn
  name='Wohnzimmer-Fenster-Lampe' (from user text)`.
- `switching_tool_names` (`[HassTurnOff, HassToggle]`): names of tools that
  change entity state and should not be run without a target. If such a call has
  no `name`, `area` or `floor`, and the current or previous user message did not
  say "alle", the call is blocked and the model receives the configured reply
  instead of executing it.
- `switching_block_reply`
  (`"Welches Gerät oder welchen Bereich meinst du genau?"`): the reply sent to
  the model when an untargeted switching call is blocked. Change it for a
  different language or wording.
- `generation_timeout` (`240`): caps how long a single generation (chat or
  completion) may run. If it is still not finished after the budget,
  the engine calls `cancel_process()`, logs a WARNING with elapsed time,
  chunk count, text chars produced so far and any tool-call fragment
  (`generation timed out after 240s: …`), and ends the stream with an
  error instead of hanging until the client gives up. `0` disables the
  budget. A client disconnecting mid-generation logs the same kind of
  summary as `generation aborted by client after …`. Raised from 120 to 240
  because on the HA host a tool call only surfaces as a chunk once fully
  generated, and prefill + tool-call generation measured 79 s to over 120 s,
  aborting valid "turn on the lamp" requests at the old budget; 240 s stays
  below Home Assistant's 300 s client timeout.
- Single-slot engine: switching models mid-flight triggers a reload.
- No request queue: concurrent requests serialize.
- No authentication: rely on HA's internal network.
- Embeddings and multi-modal are not yet supported. Function calling: see
  *Tool calling* above.
