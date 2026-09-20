# E2E — HAs LiteLLM-Integration direkt gegen das Add-on (ohne LiteLLM-Proxy)

**Datum:** 2026-09-20  **Host:** HAOS `homek` (HA Core 2026.9.2), Add-on `83680c0c_litert_llm_server` 0.5.0,
Modell `gemma-4-e2b`. Anlass: HA Core 2026.8 hat eine native **LiteLLM**-Integration (Bronze, Code-Owner `@luismalves`)
bekommen — Frage war, ob sie sich ohne LiteLLM-Proxy direkt auf unser Add-on richten lässt.

## Was die Integration tatsächlich aufruft

Gelesen in `home-assistant/core@dev`, `homeassistant/components/litellm/`:

| Aspekt | Befund |
|---|---|
| Abhängigkeit | `openai==3.10.0` — ein reiner OpenAI-Client mit `base_url`, nichts LiteLLM-Spezifisches |
| Modell-Erkennung | `client.models.list()` → `GET /v1/models`; dient zugleich als Verbindungstest im Config-Flow |
| Inferenz | `client.chat.completions.create(**model_args)`, **ohne** `stream=True` |
| Gesendete Felder | nur `model`, `messages`, `tools`, `user=conversation_id` — kein `temperature`, `max_tokens`, `timeout`, `extra_body` |
| Tools | OpenAI-Function-Format; Ergebnisse als `role="tool"` mit `tool_call_id`; Schleife max. 10 Runden |
| Rollen | `system`, `user`, `assistant`, `tool` (kein `developer`) |
| Optionen des Agenten | nur `CONF_MODEL`, `CONF_PROMPT`, `CONF_LLM_HASS_API` — **kein** Kontextfenster, **kein** Verlaufslimit |

Damit deckt sich der Bedarf vollständig mit `adapters/openai_router.py`: `/v1/models` liefert `data[].id`,
`ChatCompletionRequest` kennt `tools`/`tool_choice`, `_to_chat_turns` löst `role="tool"` über `tool_call_id` auf,
`stream` steht per Default auf `False`, und das unbekannte Feld `user` wird von Pydantic verworfen (kein 422).
Kein Add-on-Code war nötig.

## Einrichtung auf dem Host

Config-Entry `litellm` mit URL `http://83680c0c-litert-llm-server:8080` (dieselbe wie die Ollama-Integration; die
Integration hängt `/v1` selbst an), API-Schlüssel leer. Agent-Subentry auf `gemma-4-e2b`, Anweisungen = HA-Default
plus `Always call GetLiveContext with a domain, name or area filter.`, „Home Assistant steuern → Assist" an.
Ergibt die Entität `conversation.gemma_4_e2b`. Dazu eine zweite Assist-Pipeline **„litellm"** neben der bestehenden
„ollama", damit beide Wege parallel testbar bleiben; „Lokale Verarbeitung von Befehlen bevorzugen" bewusst AN
gelassen (Parität mit „ollama" → Modellpfad nur mit freier Formulierung testen).

## Messungen

| Weg | Anfrage | Tool-Call | Ergebnis | Dauer |
|---|---|---|---|---|
| Websocket `conversation/process` | „Wie warm ist es im Wohnzimmer?" | — | „The temperature in the living room is 19.0°C." | 123 s |
| Assist-Dialog, Pipeline „litellm" | „Sag mir bitte, wie warm es gerade im Wohnzimmer ist." | `homeassistant__GetLiveContext {"area": "Wohnzimmer"}` → `success` | „Die Temperatur im Wohnzimmer beträgt 19.0°C." | ~100 s |

Beide Werte decken sich mit der bekannten Referenz aus `2026-09-16-prompt-compaction-e2e.md` (19,0 °C, Sensor mit Bereich).

## Einordnung

- **Der OpenAI-Pfad trägt E2E**, inklusive Tool-Calling mit Bereichsfilter — bisher war nur der Ollama-Pfad abgenommen.
- Der Tool-Name kommt als `homeassistant__GetLiveContext`, nicht mit dem `intent__`-Präfix von `HassTurnOn`. HA
  namespaced je nach Werkzeugherkunft; dass `services/repair.py` das Präfix abstreift statt exakt zu vergleichen,
  trägt deshalb auch hier.
- Prompt-Kürzung und Tool-Call-Reparatur gelten automatisch mit, weil sie in `__main__.py` um die **Engine** gelegt
  sind und nicht im Adapter — beide Router bekommen dasselbe dekorierte Objekt.

## Offene Punkte

- Prompt-Parität mit dem Ollama-Agenten nicht verglichen: Subentry-Optionen sind über
  `config_entries/subentries/list` nicht lesbar, nur im Options-Dialog. Für ein sauberes A/B nötig.
- Die Integration überschreibt die `openai`-Client-Defaults nicht (600 s Timeout, `max_retries=2`). Läuft eine
  Generierung ins Timeout, könnte ein automatischer Retry einen zweiten Request auf den Single-Slot-Engine werfen.
  In beiden Messungen nicht eingetreten, aber ungetestet.
- `max_tokens` kommt nicht mit → es greift der Add-on-Default 512.
- Ohne Verlaufslimit geht die volle Historie pro Runde raus; die Kürzung wird auf diesem Pfad tragend statt hilfreich.
- `user=conversation_id` läge als expliziter Konversationsschlüssel bereit — mögliche Alternative zur strukturellen
  Heuristik in `engines/continuation.py`. Nicht entschieden.
