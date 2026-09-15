# Spike — litert-lm-api 0.17.0 tool-call chunk format

**Date:** 2026-09-15  **Model:** gemma-4-e2b (local, CPU)  **Script:** throwaway, not committed

## A. Streamed tool call (automatic_tool_calling=False)

Exactly one chunk arrives, then the stream ends:

```json
{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}]}
```

- Location: **top-level `tool_calls`** (no `content` key at all; no `type == "tool_call"` content items seen)
- `arguments` type: **dict**
- `id` present: **no** (the engine must generate one)
- Text before the call in the same turn: **no**

## B. Non-streamed (`send_message`)

Identical object to A.

## C. History with tool result

Accepted assistant-turn shape (top-level `tool_calls`, `content: ""`):

```json
{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Frankfurt"}}}]}
```

Accepted tool-turn shape (as built by `Conversation._handle_tool_calls`):

```json
{"role": "tool", "content": [{"type": "tool_response", "name": "get_weather", "response": {"temperature_c": 21, "condition": "sunny"}}]}
```

- Passed as `messages=` to `create_conversation` (assistant turn) and as the message to `send_message_async` (tool turn): **accepted, no error**
- Model answered using the tool result: **yes** — "The weather in Frankfurt right now is sunny with a temperature of 21°C."
- Text chunks in C have the usual shape `{"role": "assistant", "content": [{"type": "text", "text": "..."}]}`.

## Consequences for Task 5

- `_extract_tool_calls` only needs the top-level `tool_calls` branch in practice; keep the `type == "tool_call"` content-item branch as a defensive fallback (it is what `conversation.py` also checks for).
- `arguments` is already a dict; the string-coercion path is a fallback only.
- Generate `call_<hex>` ids in the engine (library gives none).
- The `_turn_to_litert` mapping in the plan is correct as written; `response` may be passed as a string (we pass the client's raw result text), the library accepted a dict here — both are JSON values.
- No text precedes the tool call, so a tool-call token can end the stream immediately.
