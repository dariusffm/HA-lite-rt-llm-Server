# TODO

## Tool-Calling für litert-llm-server (0.2.0)

Spec: `docs/superpowers/specs/2026-09-15-tool-calling-design.md`
Plan mit Schritten, Testcode und Commits: `docs/superpowers/plans/2026-09-15-tool-calling.md`

- [ ] 1. `litert-lm-api` auf 0.17.0 heben (Lock + Untergrenze), Tests + lokale Inferenz
- [ ] 2. Spike: Tool-Call-Chunk-Format von `litert_lm` gegen das lokale Modell, Ergebnis nach `docs/benchmarks/`
- [ ] 3. Domain: `ToolSpec`, `ToolCall`, `ChatTurn`/`Token` erweitern, `stream_chat(..., tools=None)`, `collect_chat` liefert Tool-Calls
- [ ] 4. Fake-Engine: skriptbare Tool-Calls, merkt sich `tools`
- [ ] 5. LiteRT-Engine: Tools an `create_conversation`, Historie mappen, Tool-Calls als Token
- [ ] 6. Ollama-Adapter: `tools`, Rolle `tool`, `message.tool_calls`
- [ ] 7. OpenAI-Adapter: `tools`, `tool_choice`, `tool_call_id`, Argumente als JSON-String, `finish_reason: tool_calls`
- [ ] 8. Option `tool_calling` (config.yaml → `LITERT_TOOL_CALLING` → Settings → Router), Log-Zeile beim Start
- [ ] 9. DOCS.md (HA-Einrichtung mit Assist und MCP-Websuche), CHANGELOG, Version 0.2.0, push
- [ ] 10. Rollout auf HA, Abnahme: Assist schaltet Gerät; MCP-Websuche beantwortet Wetterfrage; Bericht nach `docs/benchmarks/`
- [ ] 11. `simplify`-Pass, Commit, push

## Offen daneben

- [ ] Websuche-MCP-Server für HA auswählen und betreiben (Voraussetzung für Wetter/Nachrichten im Chat)
- [ ] Node-RED: Client finden oder bauen, der Tool-Calls selbst ausführt
- [ ] Vorbestehende Lint-Abweichungen: ruff I001 in `app/tests/fakes/fake_engine.py` (verschwindet mit Task 4), `ruff format` für `config.py`, `domain/*.py`, `huggingface.py` u. a.
- [ ] FastAPI-`version`-String in `__main__.py` hängt hinter `config.yaml` (wird in Task 8 auf 0.2.0 gesetzt)
