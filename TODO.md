# TODO

## Tool-Calling für litert-llm-server (0.2.0)

Spec: `docs/superpowers/specs/2026-09-15-tool-calling-design.md`
Plan mit Schritten, Testcode und Commits: `docs/superpowers/plans/2026-09-15-tool-calling.md`

- [x] 1. `litert-lm-api` auf 0.17.0 heben (Lock + Untergrenze), Tests + lokale Inferenz
- [x] 2. Spike: Tool-Call-Chunk-Format von `litert_lm` gegen das lokale Modell, Ergebnis nach `docs/benchmarks/`
- [x] 3. Domain: `ToolSpec`, `ToolCall`, `ChatTurn`/`Token` erweitern, `stream_chat(..., tools=None)`, `collect_chat` liefert Tool-Calls
- [x] 4. Fake-Engine: skriptbare Tool-Calls, merkt sich `tools`
- [x] 5. LiteRT-Engine: Tools an `create_conversation`, Historie mappen, Tool-Calls als Token
- [x] 6. Ollama-Adapter: `tools`, Rolle `tool`, `message.tool_calls`
- [x] 7. OpenAI-Adapter: `tools`, `tool_choice`, `tool_call_id`, Argumente als JSON-String, `finish_reason: tool_calls`
- [x] 8. Option `tool_calling` (config.yaml → `LITERT_TOOL_CALLING` → Settings → Router), Log-Zeile beim Start
- [x] 9. DOCS.md (HA-Einrichtung mit Assist und MCP-Websuche), CHANGELOG, Version 0.2.0, push
- [x] 10. Rollout auf HA, Abnahme: Assist schaltet Gerät; MCP-Websuche beantwortet Wetterfrage; Bericht nach `docs/benchmarks/`
- [x] 11. `simplify`-Pass, Commit, push

## Repo-Umzug in HA (direkt nach Task 10)

Repo wurde am 2026-09-15 zu `https://github.com/dariusffm/HA-lite-rt-llm-Server` umbenannt; HA hat noch die alte URL (GitHub leitet um). Slug/Hostname `83680c0c_…` ist ein Hash der Repo-URL und ändert sich beim Wechsel.
- [ ] Add-on deinstallieren (löscht `/data` inkl. Modell)
- [ ] Alte Repository-Zeile entfernen, neue URL eintragen
- [ ] Add-on neu installieren, `tool_calling` prüfen, Modell per `/api/pull` neu ziehen
- [ ] Ollama-Integration: URL auf den neuen Hostnamen ändern, Konversationsagent testen
- [ ] Neuen Hostnamen in `docs/benchmarks/` und im Projektgedächtnis nachziehen

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

- [ ] `config.yaml` `log_level` erlaubt `critical`, `config.py` `LogLevel` kennt `notice`/`fatal` statt `critical` → Auswahl `critical` in HA schlägt beim Start fehl; Werte angleichen (vorbestehend)
- [ ] `engines/litert.py`: Producer-Thread kann bei vollem Queue und abgebrochenem Client in `q.put` hängen (cancel bricht nur den Decode ab) → `put(timeout=…)` + "consumer gone"-Flag; drei `except Exception: pass` um close()/cancel() ohne Log (vorbestehend)
- [ ] `/v1/completions` ignoriert `stream: true` und antwortet immer non-stream (vorbestehend)
- [ ] Ollama `options.num_ctx` wird ignoriert (HA-Agent steht jetzt auf 16384 wie `context_length`); Warnung im Log, wenn ein Client mehr Kontext anfragt als `context_length` (Beobachtbarkeit, aus dem Simplify-Pass vertagt)
- [ ] `FakeEngine` hat drei Schalter (`tool_calls`, `raise_error`, `raise_after`) mit impliziter Priorität; beim nächsten Umbau auf ein einzelnes Skript `list[Token | Exception]` umstellen
- [ ] Modell-Katalog erweitern (Kandidaten mit Repo/Datei in `litert-llm-server/DOCS.md` → "Model Landscape"): zuerst `qwen3-0.6b` (klein, Apache-2.0) und `functiongemma-270m` (reines Tool-Routing), optional `gemma-4-12b` für starke Hosts
- [ ] Wetter im HA-Chat: Wetter-Entität (z. B. Met.no) für den Assist-Agenten freigeben; HA Core bringt ein natives Wettervorhersage-Werkzeug mit (kein Internet-Tool nötig); testen
- [ ] Nachrichten/Websuche im HA-Chat: HACS-Integration `skye-harris/llm_intents` (MIT) installieren, Backend Wikipedia oder SearXNG (kein Schlüssel) bzw. Brave (Schlüssel), Werkzeuge im Agenten freigeben; testen
- [ ] Zweiten, schlanken Konversationsagenten nur mit Wetter/News-Werkzeugen anlegen (Assist-Prompt mit allen Entitäten kostet 3–4 min pro Durchlauf)
- [ ] Später/optional MCP-Weg: SearXNG-Add-on (`pol4rfuchs/ha-apps`) + SearXNG-MCP-Server + Community-Add-on `mcp-proxy` (stdio→SSE; HAs MCP-Integration spricht nur SSE) → URL in HA-Integration "Model Context Protocol"
- [ ] Vorbestehende Format-Abweichung: `ruff format` würde ~15 Dateien umformatieren (config.py, domain/*.py, huggingface.py u. a.); bewusst nicht angefasst
