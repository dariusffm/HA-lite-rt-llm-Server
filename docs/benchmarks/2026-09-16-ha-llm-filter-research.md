# Recherche — Filter in der Home-Assistant-LLM-API (Assist)

**Datum:** 2026-09-16  **Anlass:** Kontextüberlauf durch `GetLiveContext` (alle 176 Entitäten) auf dem Test-Host (HA Core 2026.9.2).
Quelle: Quellcode-Analyse `home-assistant/core` Tag 2026.9.0 durch einen Recherche-Agenten; HA-Version des Hosts per UI geprüft.

## Befund

1. **`GetLiveContext` hat Filterparameter** (`homeassistant/components/homeassistant/llm.py`), alle optional und kombinierbar:
   `name` (Name/Alias, case-insensitive), `domain` (einzeln oder Liste, z. B. `light`), `area` (Bereichsname/Alias).
   Kein `floor`, kein `device_class`. Ohne Parameter liefert das Tool weiterhin alle exponierten Entitäten.
   Ob gefiltert wird, entscheidet allein das Modell beim Aufruf.
2. **Systemprompt nicht kürzbar.** Der statische Teil (`Static Context: An overview of the areas and the devices …`) dumpt alle exponierten Entitäten als YAML (Name, Domain, Areas, Aliase). Kein Core-Schalter für weniger Felder oder weniger Einträge; die Ollama-Integration hat ebenfalls keine Option dafür. Einziger Core-Hebel: weniger Entitäten freigeben.
3. **Custom-Alternativen (HACS, kein Core):**
   - PowerLLM (`Shulyaka/powerllm`): zusätzliche Tools über den Core-`llm`-Mechanismus, u. a. `HassGetState` für gezielte Detailzustände statt Full-Dump.
   - Extended OpenAI Conversation (`jekalmin`): eigener Agent mit frei definierbaren Functions.
   - home-llm (`acon96`): eigenes Prompt-Format, auf ~32 Entitäten ausgelegt.
4. **Ollama-Optionen** (`ollama/const.py`): `num_ctx` (Default 8192), `max_history` (Default 20, 0 = unbegrenzt), `keep_alive`, `think`. `max_history` begrenzt Nachrichten, nicht die Größe eines Tool-Ergebnisses.

## Konsequenz für das Add-on

- Die mehrstufige Abfrage braucht **kein virtuelles Tool im Add-on**: HA kann bereits nach Domäne, Name und Bereich filtern. Die Aufgabe verschiebt sich darauf, dass das Modell den Filter auch benutzt.
- Billigster Test (kein Code): in den Agent-„Anweisungen“ eine Regel wie „Rufe GetLiveContext immer mit `domain` oder `name` auf, nie ohne Filter“ ergänzen und eine Lampen-Abfrage wiederholen; im Add-on-Log prüfen, ob der Tool-Aufruf Argumente trägt.
- Falls Gemma 4 E2B den Filter trotz Anweisung weglässt: Add-on-seitig den Aufruf ergänzen ist nicht möglich (Absicht unbekannt); dann bleibt die Tool-Beschreibung im Adapter zu schärfen oder PowerLLM zu testen.
- Der Systemprompt (~6k Tokens bei 176 Entitäten) bleibt davon unberührt — dort hilft nur Kuratierung der Freigabe.
