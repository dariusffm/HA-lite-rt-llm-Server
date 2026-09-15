# TODO

## Tool-Calling für litert-llm-server (0.2.0)

Spec: `docs/superpowers/specs/2026-09-15-tool-calling-design.md`
Plan mit Schritten, Testcode und Commits: `docs/superpowers/plans/2026-09-15-tool-calling.md`

- [x] 1. `litert-lm-api` auf 0.17.0 heben (Lock + Untergrenze), Tests + lokale Inferenz
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

## Steuerung durch das Modell (nach 0.2.0)

Fall 1 — auf Zuruf im HA-Chat: abgedeckt durch Task 10 (Assist an, exponierte Entitäten).

Fall 2 — selbstständig in Automationen (Node-RED), eigene Phase mit Spec:
- [ ] Node-RED-Subflow "Tool-Schleife": Chat → Switch auf `message.tool_calls` → Tool ausführen → zweiter Chat mit Rolle `tool` (keine eigene Node; erst bei Bedarf als `node-red-contrib-litert`)
- [ ] Allowlist: die Automation bietet dem Modell nur die Tools an, die sie erlaubt (fester Entitäten-Satz), nie alle HA-Dienste
- [ ] Strukturierte Entscheidung: Tool-Schema erzwingt JSON (`aktion`, Parameter, `grund`); der Flow prüft Grenzen vor dem Schalten
- [ ] Bestätigung per Benachrichtigung (Ja/Nein) für kritische Geräte, Direktschaltung nur für unkritische
- [ ] Offene Entscheidung des Users: welche Geräteklassen darf das Modell ohne Rückfrage schalten (Licht? Steckdosen?), welche nur mit Bestätigung (Heizung, Schlösser, Tore, Rollläden)?
- [ ] Wetter/Nachrichten in Node-RED: Wetter-/News-API als Tool im Subflow anbinden

## Offen daneben

- [ ] Modell-Katalog erweitern (Kandidaten mit Repo/Datei in `litert-llm-server/DOCS.md` → "Model Landscape"): zuerst `qwen3-0.6b` (klein, Apache-2.0) und `functiongemma-270m` (reines Tool-Routing), optional `gemma-4-12b` für starke Hosts
- [ ] Websuche-MCP-Server für HA auswählen und betreiben (Voraussetzung für Wetter/Nachrichten im HA-Chat)
- [ ] Vorbestehende Lint-Abweichungen: ruff I001 in `app/tests/fakes/fake_engine.py` (verschwindet mit Task 4), `ruff format` für `config.py`, `domain/*.py`, `huggingface.py` u. a.
- [ ] FastAPI-`version`-String in `__main__.py` hängt hinter `config.yaml` (wird in Task 8 auf 0.2.0 gesetzt)
