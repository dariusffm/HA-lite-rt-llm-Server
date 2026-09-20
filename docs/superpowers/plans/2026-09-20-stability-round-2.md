# Plan — Stabilitätsrunde 2 (nach 0.5.1)

**Stand:** `main` = `24eb5ed` (0.5.1). Der Patch `HA-LiteRT-0.5.1-stability.patch` ist eingespielt —
Engine-Zustand nach fehlgeschlagenem Wechsel, Cache-Trennung je Modell, LRU, `HassTurnOn` im
Python-Default sind **erledigt und nicht erneut zu implementieren**.

**Entscheidungen des Nutzers (2026-09-20):**
- Zwei Branches: **A** = Punkte 3/4/5 (lokal beweisbar), **B** = Punkte 1/2 (Nebenläufigkeit, nur auf
  HA-Hardware wirklich verifizierbar).
- Zugriffskontrolle: **strikte Serialisierung** — ein Slot, Freigabe erst nach bestätigtem Ende des
  nativen Producer-Threads.
- Bei belegter Engine: **warten mit Deckel**, danach sauberes 503.

Kein Auto-Merge, kein Rollout auf HA. Alle Pfade relativ zu `litert-llm-server/`.

---

## Branch A — `fix/config-wiring-and-completion-stream`

### A1 · HF_TOKEN erreicht s6 nicht (Punkt 3, hoch)

`rootfs/etc/cont-init.d/01-config.sh:42` filtert `printenv` mit `^(LITERT_|HF_TOKEN$)`. `printenv`
gibt `HF_TOKEN=wert` aus; das `$` verankert direkt hinter `HF_TOKEN` und trifft diese Zeile nie.
Folge: der Token landet nicht in `/run/s6/container_environment/`, der uvicorn-Dienst startet ohne
ihn, gated `google/*-litert-lm`-Downloads scheitern mit 401.

- Filter auf `^(LITERT_|HF_TOKEN=)` korrigieren.
- Test: `bash`-Test mit künstlichem Token (`hf_TESTONLY…`), der die Schleife isoliert ausführt und
  prüft, dass die Datei entsteht. **Nie** echte Tokenwerte ausgeben oder loggen; der Test darf den
  Wert nur gegen seinen eigenen Fixture-Wert vergleichen.
- Risiko: Die Schleife läuft in einer Subshell (`| while`), Zuweisungen wirken nicht nach außen —
  hier unkritisch, da nur Dateien geschrieben werden.

### A2 · Settings erreichen die Anwendung nicht (Punkt 4, mittel)

Bestätigt per grep: `default_model` und `preload_models` kommen außerhalb von `config.py` und
`tests/test_config.py` **nirgends** vor; `settings.max_tokens`/`temperature` nur in der
Warnung `__main__.py:147`.

1. **`default_model`** → als `default_model` an beide Router-Builder; `model` in den Request-Modellen
   wird optional und fällt auf den Standard zurück. Explizite Client-Werte haben Vorrang.
2. **`max_tokens` / `temperature`** → als Router-Defaults injizieren statt der hartkodierten 512/0.7
   in `openai_router.py`. Pydantic-Defaults durch `None` ersetzen und im Router auflösen, damit
   „Client hat nichts gesagt" von „Client hat 512 gesagt" unterscheidbar bleibt.
3. **`preload_models`** → in `make_production_app` beim Start über die Registry ziehen.
   Nicht blockierend beim Import, Fehler werden geloggt, nicht geworfen (ein fehlender Download darf
   den Dienst nicht am Start hindern).
4. **`params.max_tokens` im Chatpfad** → `create_conversation` akzeptiert laut Signatur von
   litert-lm-api 0.17 `max_output_tokens: int | None`; bisher wird es nur in `stream_completion`
   über `create_session` gesetzt. Durchreichen.
   **Achtung, Folgeentscheidung:** `engines/continuation.py:31` schließt `max_tokens` bewusst aus dem
   `config_key` aus („per call"). Das Limit wird aber bei der *Erstellung* der Conversation gesetzt —
   eine wiederverwendete Conversation erbt damit still das Limit des ersten Aufrufs.
   → `max_tokens` in die Fortsetzungsentscheidung aufnehmen (abweichender Wert = keine
   Wiederverwendung). Kosten: ein Client, der pro Runde andere `max_tokens` schickt, verliert die
   Wiederverwendung. HAs Ollama- und LiteLLM-Pfad schicken pro Konversation einen konstanten Wert,
   also praktisch folgenlos.

Architekturgrenze: konkrete Settings **nur** in `make_production_app` verdrahten, `build_app` bleibt
über Parameter injizierbar (CLAUDE.md: `__main__` ist der einzige Ort für konkrete Instanzen).

### A3 · `/v1/completions` ignoriert `stream` (Punkt 5, mittel)

`adapters/openai_router.py:294` ruft immer `collect_completion()`. Das SSE-Framing für Chat existiert
im selben Modul bereits (`sse_stream`, `_error_frame`) — Muster übernehmen, nicht neu erfinden:
Deltas als `text`-Feld des Completion-Schemas, Abschluss mit `finish_reason`, dann `data: [DONE]`.
Fehler nach Teilausgabe gehen als Fehler-Frame in den laufenden Stream (HTTP-Status ist da schon weg).

**Tests A:** Filter-Test (A1); je ein Test für Default-Übernahme und Client-Vorrang bei `model`,
`max_tokens`, `temperature`; Preload ruft die Registry; `max_output_tokens` erreicht
`create_conversation`; abweichende `max_tokens` unterbinden die Wiederverwendung; Completion-Stream
normal, Fehler nach Teilausgabe, Client-Abbruch.

**Version:** 0.6.0 (`config.yaml` + FastAPI-`version` + CHANGELOG gemeinsam) — `model` wird optional
und Adapter-Defaults ändern sich, das ist mehr als ein Patch-Release.

---

## Branch B — `fix/engine-serialization` (nach A)

### B1 · Gemeinsame Zugriffskontrolle (Punkt 1, hoch)

Heute schützt `self._lock` (ein `threading.Lock`) in `engines/litert.py:399` nur `_ensure_loaded`.
`stream_chat`, `stream_completion` und die Stufe-1-Kürzung laufen danach ungeschützt auf derselben
Engine; ein Modellwechsel kann `close()` aufrufen, während ein anderer Producer-Thread noch dekodiert.

Entwurf: **ein `asyncio.Semaphore(1)` als Engine-Gate**, erworben in den beiden öffentlichen
`async`-Einstiegen, freigegeben erst, wenn der native Producer sein Ende bestätigt hat
(`producer_done`-Event bzw. Thread-Ende), nicht schon beim Verlassen des Python-Generators.
- Das Gate lebt **in der Engine**, nicht als `services/`-Dekorator: es braucht Wissen über den
  nativen Thread, das die Domain-Protokolle nicht kennen (CLAUDE.md: `services/` importiert nur
  `domain/`).
- Erwerb mit Deckel: neue Option `engine_wait_timeout` (Default 60 s, 0 = unbegrenzt) →
  Überschreitung ergibt eine Domain-Ausnahme, die die Adapter als 503 abbilden.
- Bestätigt der Producer sein Ende **nicht** innerhalb einer Frist, wird das Gate **nicht**
  freigegeben und die Engine als unbrauchbar markiert; Folgeanfragen bekommen 503 statt auf einer
  halb geschlossenen Engine weiterzulaufen. `/readyz` spiegelt das.
- Stufe 1 der Kürzung und der Hauptaufruf laufen dadurch nacheinander. Erwartete Folge auf HA: die
  Gesamtdauer bleibt gleich (sie teilten sich ohnehin einen Slot), aber die Verschränkung
  verschwindet.

### B2 · Abbruch und Bereinigung (Punkt 2, hoch)

- `_fire_cancel` (`litert.py:130`) macht `thread.join(0.1)` **auf dem Event-Loop** → bis zu 100 ms
  blockiert der gesamte Server. In einen Daemon-Thread verlagern bzw. `await asyncio.to_thread`.
- `_drop_held` (`:423`) ruft `conversation.close()` synchron auf dem Loop — dieselbe Behandlung.
- Verschachtelte `async for`-Weiterleitungen (Adapter → Service-Dekoratoren → Engine): beim
  vorzeitigen Schließen des äußeren Streams ist nicht garantiert, dass alle inneren Generatoren
  sofort finalisiert werden. → explizite `aclose()`-Kette statt auf GC zu vertrauen.
- Reihenfolge Tool-Call-Ausgabe vs. Producer-Abschluss prüfen.

**Tests B:** zwei gleichzeitige Chats; Chat + Completion; Modellwechsel während laufender Inferenz;
neue Anfrage unmittelbar nach Timeout; Client-Abbruch; langsames `cancel`; langsames `close`;
Timeout ohne Ausgabe; Abbruch nach einem Tool-Call. Alle mit `FakeEngine`-Doubles, die die
Verzögerungen simulieren — der Event-Loop muss dabei ansprechbar bleiben (messbar über eine
parallele Heartbeat-Task).

**Grenze der Aussagekraft:** Diese Tests beweisen die *Python-seitige* Serialisierung und
Loop-Reaktivität. Sie beweisen **nicht**, dass die nativen Threads in litert-lm frei von Races sind
oder dass Hardware-Timeouts verschwinden. Das gilt erst nach einem Lauf auf dem HA-Host als geprüft
und wird bis dahin ausdrücklich als offen berichtet.

---

## Abnahme (beide Branches)

Im Verzeichnis `litert-llm-server/app`: `uv run pytest`, `uv run ruff check .`, `uv run mypy src/`,
zusätzlich `uv run lint-imports` (die vier Architekturverträge). CHANGELOG und `TODO.md` je Branch
aktualisieren. Kein Merge, kein Rollout.

---

# Revision 1 — nach Architektur- und Security-Review (2026-09-20)

Beide Reviews liefen gegen die Fassung oben. **Wo diese Revision dem Text oben widerspricht, gilt
diese Revision.** Die zitierten Befunde habe ich jeweils am Code nachgeprüft, nicht übernommen.

## Neu in Branch A, vor allem anderen

### A0 · Modellnamen validieren (Security-Befund C1, nicht im Auftrag enthalten)

`adapters/ollama_router.py:374` exponiert `DELETE /api/delete` → `registry.delete(req.name)` →
`model_registry/filesystem.py:38` `self.root / f"{name}{MODEL_EXT}"` — ohne Prüfung, ohne Auth, Port
8080 im LAN veröffentlicht (`config.yaml:11`). Dieselbe Kette in `engines/litert.py:396`
`_model_file`. Schadensgrenze: es lassen sich nur Pfade treffen, die auf `.litertlm` enden — real,
aber kleiner als „beliebige Dateien".

- Ein Validator an **einer** Stelle (`domain/`): `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`.
- Zusätzlich `Path.resolve()` + `is_relative_to(models_dir)` in `path_for` **und** `_model_file`
  (Gürtel und Hosenträger, weil Symlinks die Regex allein aushebeln).
- Vor A2 umsetzen: A2 speist `default_model` und `preload_models` zusätzlich in diese Pfade.
- Nutzerentscheidung: in Branch A mitnehmen, kein eigener Branch.

## Änderungen an A1

- `umask 077` vor der Persist-Schleife (W1): heute entsteht die Tokendatei mit 0644.
- `hf_token` als `SecretStr` in `config.py`, `repr=False` im Registry-Feld (W2) — sonst druckt ein
  `log.debug("%r", settings)` den Token in den HA-Log. Test: Token taucht in `repr()` nicht auf.
- Test hermetisch: `env -i`/`unset HF_TOKEN` am Eingang, Zielverzeichnis aus `mktemp -d` mit
  `trap cleanup EXIT`, Vergleich über Bool/sha256 statt `echo`/`diff`, kein `set -x`. Fixture-Wert
  `hf_TESTONLY_not_a_real_token` — erfüllt bewusst **nicht** das echte HF-Format, sonst schlägt
  Secret Scanning an, wenn das Repo öffentlich wird.
- **Negativtest:** ohne gesetztes `HF_TOKEN` darf die Datei nicht entstehen, sonst kippt der Fix die
  bewusste „blank ⇒ unset"-Semantik aus `01-config.sh:33-38`.

## Änderungen an A2

- **`ollama_router.py:112-113` hat dieselben harten `512`/`0.7`** wie der OpenAI-Adapter (im
  Ursprungsplan übersehen). Beide Adapter ziehen, beide Pfade testen.
- Defaults als **ein** eingefrorenes `RouterDefaults`-Objekt (`default_model`, `max_tokens`,
  `temperature`, `tools_enabled`) statt vier weiterer Skalar-Parameter an beiden Buildern.
- `model` wird optional → **einmal ganz oben im Handler auflösen** und nur die lokale Variable
  weiterreichen, auch in Antwort und Logzeilen. Sonst echot die Antwort `null`, während die Engine
  gegen `default_model` lief. Eigener Test: „Antwort nennt das aufgelöste Modell".
- `preload_models` **nicht** im Factory-Body: `make_production_app` ist synchron und läuft vor dem
  Socket-Bind — ein Download dort macht `/healthz` unerreichbar und HA meldet Startfehler. Stattdessen
  `build_app(..., on_startup=…)` als Lifespan-Hook, Closure aus `__main__`.
- `HuggingFaceRegistry.pull` **wirft nicht**, sondern yieldet `PullProgress(status="error", …)`
  (`huggingface.py:139`). Den letzten Fortschritt auswerten und `status == "error"` mit `prog.error`
  loggen — sonst schluckt ein `async for _ in …: pass` den Fehler lautlos. Testkriterium.
- `max_tokens` im Chatpfad: `create_conversation` kennt `max_output_tokens`. Der Review hält das
  Aufnehmen in `config_key` für Symptombehandlung und schlägt vor, das Limit stattdessen im
  Engine-Consumer durchzusetzen (nach N Token abbrechen, `finish_reason="length"`), damit der
  Domänenvertrag „per call" wahr wird und die Wiederverwendung maximal bleibt.
  → **Entscheidung: Consumer-seitige Durchsetzung.** `config_key` und sein Docstring bleiben
  unangetastet, `create_conversation` bekommt **kein** `max_output_tokens`.

## Änderungen an A3

- Fehlermapper in beiden Adaptern: nach außen feste Meldung + opake `error_id` (uuid4), Details nur
  via `log.exception` mit derselben id. Heute leakt `_ensure_loaded` den absoluten Containerpfad
  (`FileNotFoundError(f"Model file not found: {file}")`) — zusammen mit A0 ein Dateisystem-Orakel.
  Stacktraces gehen schon heute **nicht** über die Leitung, es geht um Meldungen und Pfade.
- Test „Fehler nach Teilausgabe" muss konkret verlangen, dass vor `data: [DONE]` ein terminaler Frame
  mit unterscheidbarem `finish_reason` steht.

## Branch B — Reihenfolge und Entwurf geändert

1. **B2 vor B1.** Sitzt die Semaphor-Freigabe im `finally` eines Generators, den niemand `aclose()`t,
   leckt der einzige Slot beim Stufe-1-Timeout und der Dienst antwortet dauerhaft 503. Die
   `aclose()`-Kette ist **Voraussetzung** für das Gate, kein Nebenpunkt. Test: „Stufe-1-Timeout →
   nächster Request bekommt den Slot".
2. **Gate umspannt die Anfrage, nicht den Aufruf.** `services/compaction.py:159` zieht Stufe 1 über
   `collect_chat(self._inner, …)` vollständig leer, bevor der Hauptaufruf startet — zwei Erwerbe,
   dazwischen kann ein fremder Request das Modell wechseln. Lösung: explizite
   `async with engine.session()`-Klammer, die `__main__` den Dekoratoren injiziert.
3. **Gate als eigenes Modul `engines/gate.py`** ohne `litert_lm`-Import, in `LiteRTEngine`
   komponiert. Sonst ist es von den `FakeEngine`-Doubles gar nicht ausgeführt und die B-Tests
   beweisen nichts — und `litert.py` (771 Zeilen) driftet weiter zur God Class.
4. **`prime()` in `domain/`**: `StreamingResponse` wird gebaut, bevor der erste `__anext__` läuft —
   ein 503 kann sonst nie als Status ankommen. Beide Adapter ziehen den ersten Token vor dem Bau der
   Response; Gate-Fehler werden damit 503, spätere Fehler bleiben Fehler-Frame.
5. **`domain/errors.py`** neu: `EngineBusyError` (Deckel überschritten) und `EngineUnavailableError`.
   Adapter fangen beide **namentlich** vor dem generischen `except Exception` und antworten 503 mit
   `Retry-After`. `services/compaction.py:164` (`except Exception → _Skip`) schluckt beide für
   Stufe 1 — das ist dort richtig und wird als bewusste Entscheidung festgehalten.
6. **TTL-Timer unter das Gate**: `_expire_held` (`litert.py:474`) prüft nur `held.busy`, das nur im
   Wiederverwendungspfad gesetzt wird. Der Timer darf nicht direkt schließen, sondern reiht eine Task
   ein, die das Gate erwirbt. (Einordnung: der Timer schließt die *gehaltene*, nicht die aktive
   Conversation — das Risiko liegt auf nativer Ebene und ist plausibel, aber unbewiesen. Wird
   abgesichert, nicht als belegter Bug behandelt.)
7. **`producer_done` nach `_bridge_producer` hochziehen** — heute gibt es das Event nur in
   `stream_chat`; `stream_completion` hat kein Abschlusssignal, die Freigaberegel wäre dort nicht
   implementierbar.
8. **Wartedeckel = `generation_timeout`** (Default 240 s, nicht 60 s): ein Request hält den Slot bis
   zu `generation_timeout`, ein 60-s-Deckel würde im Normalfall 503 liefern. Dazu maximale
   Warteschlangentiefe 4 (darüber sofort 503) und nach dem Erwerb prüfen, ob der Anfragende noch da
   ist. Neue Option braucht vier Touchpoints: `config.yaml` (Option **und** Schema `int(0,600)`),
   `01-config.sh`, `config.py`, `make_production_app`.
9. **Hängender Producer → Prozessende.** `/readyz` auf `false`, `log.error` mit Grund, dann
   kontrolliertes `sys.exit(1)`; s6 startet neu. Ein verhängter nativer Thread ist prozessintern
   nicht zurückzugewinnen — „dauerhaft 503" wäre ein Ausfall ohne Selbstheilung.
   `/readyz` braucht dafür `build_app(..., engine_ok=…)` mit Referenz auf die **undekorierte**
   Engine; das `InferenceService`-Protocol wird **nicht** erweitert.
10. **Semaphore statt Lock** ist richtig, weil die Freigabe aus einem anderen Kontext kommt als der
    Erwerb (Producer-Bestätigung per `call_soon_threadsafe`) — ohne diese Begründung wird das beim
    nächsten Durchgang zu `async with lock` „vereinfacht".
11. **Version Branch B: 0.7.0** (neues Antwortverhalten 503 + neue Option), eigener gemeinsamer Bump.

## Bewusst nicht übernommen

- **Auth / Ingress (R7), Revision-Pinning (R3), atomarer Modelltausch (R4), Plattenplatzprüfung
  (R5):** echte Befunde, aber außerhalb der fünf beauftragten Punkte und je für sich ein eigener
  Schnitt. Gehen als Einträge nach `TODO.md`, nicht in diese Branches.
- **`log_level: trace` druckt womöglich den Token (W3):** lässt sich nur auf dem HA-Host mit einem
  Wegwerf-Token verifizieren. Als Abnahmeschritt in `TODO.md`, nicht als Codeänderung geraten.
