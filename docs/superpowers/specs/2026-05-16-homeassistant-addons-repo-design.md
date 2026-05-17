# Design — Home Assistant Add-on Repository mit LiteRT LLM Server

**Status:** Draft (zur User-Freigabe)
**Datum:** 2026-05-16
**Scope:** Repo-Aufsetzen + erstes Add-on `litert-llm-server`

---

## 1. Ziel

Ein Home-Assistant-Add-on-Repository aufsetzen, in dem mehrere Add-ons in
Unterordnern entwickelt werden. Das erste Add-on, `litert-llm-server`,
stellt ein lokales LLM via Google LiteRT (ehem. TensorFlow Lite) bereit
und exponiert es über zwei parallele HTTP-APIs (OpenAI-kompatibel und
Ollama-kompatibel), damit Home Assistant und Node-RED es ohne
Custom-Integrationen nutzen können.

Distribution ist zunächst rein lokal. Public Github + GHCR-Pre-Built-Images
sind als Phase 2 vorgesehen — die Repo-Struktur ist von Beginn an
HA-Add-on-Repository-konform, damit der Übergang ein reines Hinzufügen
von CI-Workflows ist.

## 2. Harte Anforderungen

| Aspekt | Entscheidung |
|---|---|
| Repo-Form | HA Add-on Repository, mehrere Add-ons in Unterordnern |
| Repository-Manifest | `repository.yaml` im Repo-Root |
| Erstes Add-on | `litert-llm-server` |
| Inferenz-Engine | Google LiteRT-LM (via `litert-lm-api`) |
| Unterstützte Modelle (MVP) | Mehrere parallel auswählbar: Gemma 4 E2B (default), Gemma 4 E4B, Gemma 3n E2B, Gemma 3n E4B. Alle als offizielle `.litertlm`-Builds — `litert-community/*` (nicht gated, kein Token nötig) und `google/*` (gated, HF-Token erforderlich). |
| Modell-Lifecycle | Auto-Download via Ollama-kompatibler `/api/pull`-Endpoint; Cache unter `/data/models/` |
| HTTP-APIs | OpenAI-kompatibel **und** Ollama-kompatibel parallel — beide aktiv |
| Streaming | SSE (OpenAI) und NDJSON (Ollama) |
| Architekturen | `amd64` + `aarch64` (x86_64 NUC/Server **und** ARM64 Pi 4/5 / ARM-NUCs / Apple-Silicon-Docker). armv7/armhf/i386 sind out-of-scope (`litert-lm-api`-Wheels nicht verfügbar; LLM-Inferenz auf 32-bit ARM unbrauchbar). |
| HuggingFace-Auth | Erforderlich für `google/*-litert-lm`-Repos (Gemma 3n): diese sind gated. `litert-community/*-litert-lm`-Repos (Gemma 4) sind public. Add-on liest `HF_TOKEN` aus ENV (von bashio aus `config.yaml`-Option `hf_token` exportiert) — optional, nur für gated Repos nötig. |
| Distribution | Lokal jetzt, Public Github + GHCR später (Struktur jetzt kompatibel) |
| Eigner-Profil | Solo-Dev, MVP-fokussiert, schnelle Iteration |

## 3. Architektur-Entscheidung

### Gewählt: Python + FastAPI + `litert-lm-api`

Begründung (Konsens zweier unabhängiger Architektur-Reviews):

- **Geringste accidental complexity**: FastAPI ist ein dünner HTTP-Layer
  ohne Framework-Lock-in. Hexagonale Architektur lässt sich sauber abbilden.
- **Klare Bounded Contexts**: Zwei API-Router (OpenAI, Ollama) sind
  Protocol Adapters, die einen einzigen `InferenceService` ansprechen —
  Engine-Wechsel ist später ohne Adapter-Änderung möglich.
- **Standard-Python-Stack**: `litert-lm-api` (PyPI) ist Googles
  produktionsreifer Python-Wrapper über LiteRT-LM (dem aktuellen
  Inference-Framework, das die deprecated MediaPipe-LLM-Inference-API
  abgelöst hat). Bietet `Engine`/`Session`/`Conversation`/`Benchmark`/
  `SamplerConfig`/`Tool` als sauberes API; kein Bazel/Custom-Build nötig.
- **Iterationsgeschwindigkeit**: Lokales Testen ohne Docker möglich
  (`uv run pytest`), schnelle Inner-Loops.

> **Historische Anmerkung (2026-05-17)**: Die ursprüngliche Spec
> referenzierte `ai-edge-litert` + `mediapipe-genai` als Inferenz-Stack.
> Beim ersten Implementierungs-Schritt zeigte sich, dass
> `mediapipe.tasks.python.genai.inference.LlmInference` in den aktuellen
> PyPI-Distributionen nicht mehr existiert und die API als deprecated
> markiert ist. Google hat den LLM-Inferenz-Pfad in das separate Paket
> `litert-lm-api` ausgelagert. Die Ports-&-Adapters-Architektur war
> robust gegenüber diesem Wechsel — nur die `LiteRTEngine`-Implementierung
> ist betroffen.

### Verworfen

- **LiteServe (Lightning AI)**: Bringt Batching/Worker-Features mit, die im
  Solo-Use-Case-NUC-Szenario nutzlos sind. OpenAI/Ollama-Schemas müssten
  zusätzlich manuell nachgebaut werden — Lock-in ohne Mehrwert.
- **C++ MediaPipe + dünner Python-Wrapper**: Bazel-Build im Dockerfile
  (Image-Build 20–40 Min.), komplexe Toolchain. Productivity-Killer für
  Solo-Dev-MVP.

### Bekanntes Risiko (vom Advisor-Review hervorgehoben)

**LiteRT-Performance auf CPU-only amd64 ist drastisch hinter dem
Marketing.** Auf x86 ohne GPU-Delegate läuft LiteRT auf XNNPACK. Realistisch
zu erwartende Token-Raten für Gemma-4-E2B (int4) liegen im einstelligen
bis niedrig-zweistelligen tok/s-Bereich. Größere Varianten (E4B) können
auf CPU schon grenzwertig sein.

Zum Vergleich: `llama.cpp` mit GGUF-Quants schafft auf identischer
Hardware 15–30 tok/s für vergleichbar große Modelle.

**Konsequenz für die Implementierung**: Der **erste Implementierungs-Step
ist ein PoC-Benchmark** mit Gemma-4-E2B auf der Ziel-Hardware. Der
Benchmark nutzt `litert_lm.Benchmark` und reportet
`last_decode_tokens_per_second` als das maßgebliche Akzeptanzkriterium.
Ergebnis < 5 tok/s → Engine-Tausch (z. B. `llama-cpp-python`) wird
erwogen, bevor der API-Layer ausgebaut wird. Die Ports-&-Adapters-Architektur
erlaubt diesen Tausch ohne Eingriff in Router oder API-Schemas.

**MVP-Modellliste (Stand 2026-05-17, korrigiert nach Smoke-Test)**

| Logischer Name | HF-Repo | Datei | Gated | Default? |
|---|---|---|---|---|
| `gemma-4-e2b` | `litert-community/gemma-4-E2B-it-litert-lm` | `gemma-4-E2B-it.litertlm` | nein | ✓ |
| `gemma-4-e4b` | `litert-community/gemma-4-E4B-it-litert-lm` | `gemma-4-E4B-it.litertlm` | nein |  |
| `gemma-3n-e2b` | `google/gemma-3n-E2B-it-litert-lm` | `gemma-3n-E2B-it-int4.litertlm` | ja (HF-Token erforderlich) |  |
| `gemma-3n-e4b` | `google/gemma-3n-E4B-it-litert-lm` | `gemma-3n-E4B-it-int4.litertlm` | ja (HF-Token erforderlich) |  |

Aus dem MVP gestrichen / nicht (mehr) relevant:
- Gemma 2B / Gemma 3-1B / Gemma 3-4B (alte Modellgeneration, deren `google/gemma-X-Y-tflite`-Repos teilweise retired sind; Files-Format `.task` ist Vergangenheit)
- Phi-2 (keine offiziellen LiteRT-LM-Builds vorhanden)

**Aufgelöstes Risiko (2026-05-17)**: Die Sorge, dass MediaPipe keine
echte Per-Token-Streaming-API biete und ein Whitespace-Chunking-Workaround
nötig wäre, ist mit dem Wechsel auf `litert-lm-api` **hinfällig** —
`litert_lm.Session` liefert nativ Token für Token aus dem Decode-Loop.

### Modell-Kontextfenster

Die LiteRT-LM-Modelle (Gemma 4 und Gemma 3n) unterstützen ein
Kontextfenster von **bis zu 32k Tokens** (Prompt + Completion zusammen).
Konsequenz für das Add-on-Schema:

- `max_tokens` Upper Bound: **32768** (statt der ursprünglich angedachten
  8192). Gilt sowohl für die HA-`config.yaml`-Option als auch für die
  Request-Parameter in `/v1/chat/completions` und `/api/chat`.
- Die effektive Output-Länge eines einzelnen Calls = `32768 − Prompt-Token`.
  Wird das überschritten, schneidet `litert_lm.Session` ab oder wirft
  einen Fehler; die Adapter sollen Tokens dann mit `finish_reason="length"`
  beenden (statt 500).
- Lange Multi-Turn-Conversations können das Kontextfenster aufzehren; ein
  Truncation-Policy (älteste Messages droppen) ist out-of-scope MVP und
  Aufgabe der `__main__.py`-Wiring im Folge-Refactor.

## 4. Repository-Layout

```
homassist-addons/
├── repository.yaml             # HA-Repo-Manifest
├── README.md                   # Repo-Übersicht, Installations-URL (Custom Repo)
├── CLAUDE.md                   # Konventionen für Claude Code (siehe Sektion 7)
│
├── litert-llm-server/          # Erstes Add-on
│   ├── config.yaml             # HA-Add-on-Schema (Ports, Options, Schema)
│   ├── Dockerfile              # FROM ghcr.io/hassio-addons/base-python:14.0.2
│   ├── build.yaml              # Build-Args, OCI-Labels
│   ├── README.md               # Nutzer-Doku (Install, Optionen)
│   ├── DOCS.md                 # Detail-Doku (API, Modelle, Limits)
│   ├── CHANGELOG.md            # Versionen
│   ├── icon.png / logo.png     # Platzhalter zunächst
│   │
│   ├── rootfs/                 # Wird ins Container-Image gemerged
│   │   └── etc/
│   │       ├── services.d/litert/
│   │       │   ├── run         # s6-Service: startet Uvicorn
│   │       │   └── finish      # s6-Cleanup
│   │       └── cont-init.d/
│   │           └── 01-config.sh  # bashio: config.yaml → ENV
│   │
│   └── app/                    # Python-Applikation
│       ├── pyproject.toml      # uv/pip dependencies
│       ├── src/litert_server/
│       │   ├── __main__.py     # FastAPI/Uvicorn-Entry, Wiring
│       │   ├── config.py       # ENV → typed Settings (pydantic-settings)
│       │   ├── domain/         # Engine-agnostisch
│       │   │   ├── inference.py        # InferenceService Protocol
│       │   │   ├── model_registry.py   # ModelRegistry Protocol
│       │   │   └── types.py            # Token, GenerationParams, ModelInfo
│       │   ├── engines/
│       │   │   └── litert.py           # LiteRTEngine
│       │   ├── adapters/
│       │   │   ├── openai_router.py    # /v1/* — SSE
│       │   │   └── ollama_router.py    # /api/* — NDJSON
│       │   └── model_registry/
│       │       ├── filesystem.py       # Cache-Verzeichnis-Logik
│       │       └── huggingface.py      # Downloader
│       └── tests/
│           ├── adapters/
│           ├── engines/
│           └── domain/
│
└── docs/                       # Repo-weite Doku
    └── superpowers/specs/      # Design-Specs (dieses Dokument)
```

`.github/workflows/` wird in Phase 2 (Public Github) hinzugefügt, nicht jetzt.

## 5. Add-on-Skelett (Dateiinhalte)

### `litert-llm-server/config.yaml`

```yaml
name: "LiteRT LLM Server"
version: "0.1.0"
slug: litert_llm_server
description: "Local LLM inference via Google LiteRT with OpenAI- and Ollama-compatible APIs"
arch:
  - amd64
  - aarch64
init: false
startup: application
boot: auto
ports:
  8080/tcp: 8080
ports_description:
  8080/tcp: "HTTP API (OpenAI + Ollama)"
options:
  log_level: info
  default_model: "gemma-4-e2b"
  max_tokens: 1024
  temperature: 0.7
  preload_models: []
  hf_token: ""
schema:
  log_level: list(trace|debug|info|warning|error|critical)
  default_model: str
  max_tokens: int(1,32768)
  temperature: float(0.0,2.0)
  preload_models:
    - str
  hf_token: password?
```

> **Anmerkung zur `log_level`-Liste**: uvicorn akzeptiert
> `critical|error|warning|info|debug|trace`. Die Schema-Enum spiegelt
> exakt diese Werte; ein bashio-Wert wird ungemappt an `uvicorn
> --log-level` durchgereicht. Die HA-typischen Werte `notice` und
> `fatal` sind bewusst ausgeschlossen, weil uvicorn sie mit
> "Invalid log level" abweisen würde.

> **Anmerkung zu `hf_token`**: HA-Add-on-Schema-Typ `password?` ist
> optional und wird in der HA-UI maskiert dargestellt. Der Wert wird via
> bashio als `HF_TOKEN` ENV-Variable exportiert — die Variable, die
> `huggingface_hub` standardmäßig liest. Ohne Token sind Gemma-Modelle
> nicht abrufbar (gated repository).

### `litert-llm-server/Dockerfile`

```dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

ARG BUILD_ARCH

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apk add --no-cache libstdc++ libgomp

COPY app/ /opt/app/
WORKDIR /opt/app

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache .

COPY rootfs/ /

LABEL io.hass.version="0.1.0" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
```

### `litert-llm-server/build.yaml`

```yaml
build_from:
  amd64: ghcr.io/hassio-addons/base-python:14.0.2
  aarch64: ghcr.io/hassio-addons/base-python:14.0.2
labels:
  org.opencontainers.image.source: "https://github.com/USER/homassist-addons"
```

(Der Platzhalter `USER` wird beim Übergang zu Public Github ersetzt.)

### `rootfs/etc/cont-init.d/01-config.sh`

```bash
#!/usr/bin/with-contenv bashio
set -euo pipefail

bashio::log.info "Reading add-on configuration..."

export LITERT_LOG_LEVEL="$(bashio::config 'log_level')"
export LITERT_DEFAULT_MODEL="$(bashio::config 'default_model')"
export LITERT_MAX_TOKENS="$(bashio::config 'max_tokens')"
export LITERT_TEMPERATURE="$(bashio::config 'temperature')"
export LITERT_MODELS_DIR="/data/models"
export LITERT_PORT="8080"

LITERT_PRELOAD_MODELS="$(bashio::config 'preload_models')"
export LITERT_PRELOAD_MODELS

# HuggingFace token: exported as HF_TOKEN only when the operator set a
# non-empty value (the env var huggingface_hub reads by default).
if bashio::config.has_value 'hf_token'; then
    export HF_TOKEN="$(bashio::config 'hf_token')"
fi

# Persist for s6-overlay v3 (base-python:14.0.2 ships v3).
mkdir -p /run/s6/container_environment
printenv | grep -E '^(LITERT_|HF_TOKEN$)' | while IFS='=' read -r key value; do
    printf '%s' "${value}" > "/run/s6/container_environment/${key}"
done
```

### `rootfs/etc/services.d/litert/run`

```bash
#!/usr/bin/with-contenv bashio
cd /opt/app
exec uvicorn --factory litert_server.__main__:make_production_app \
    --host 0.0.0.0 \
    --port "${LITERT_PORT}" \
    --log-level "${LITERT_LOG_LEVEL}"
```

### `rootfs/etc/services.d/litert/finish`

```bash
#!/command/execlineb -S1
s6-svscanctl -t /run/service
```

## 6. Applikations-Architektur

### Modul-Graph

```
__main__.py  (FastAPI app, Uvicorn entry)
   │
   │   Wires concrete Engine + Registry + Routers
   │
   ├── adapters/openai_router.py  ─┐
   │                                 │
   └── adapters/ollama_router.py  ─┤  ADAPTERS (Protocol)
                                     │
                                     ▼
                       domain/inference.InferenceService  (Protocol)
                       domain/model_registry.ModelRegistry  (Protocol)
                                     ▲
                                     │
                    Implementations: │
                                     │
   engines/litert.py  ───────────────┤   APPLICATION/DOMAIN
   model_registry/huggingface.py  ───┘
   model_registry/filesystem.py
```

### Schnittstellen (Skizzen)

```python
# domain/types.py
class GenerationParams(BaseModel):
    max_tokens: int
    temperature: float
    top_p: float | None = None
    stop: list[str] | None = None

class Token(BaseModel):
    text: str
    index: int
    finish_reason: Literal["stop", "length", None] = None

class ModelInfo(BaseModel):
    name: str            # "gemma-4-e2b"
    size_bytes: int
    quantization: str    # "int4", "int8", "fp16"
    path: Path | None    # None = nicht lokal vorhanden
```

```python
# domain/inference.py
class InferenceService(Protocol):
    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]: ...
```

```python
# domain/model_registry.py
class PullProgress(BaseModel):
    bytes_done: int
    bytes_total: int
    status: Literal["downloading", "verifying", "done", "error"]
    error: str | None = None

class ModelRegistry(Protocol):
    async def list(self) -> list[ModelInfo]: ...
    async def get(self, name: str) -> ModelInfo: ...
    async def pull(self, name: str) -> AsyncIterator[PullProgress]: ...
    async def delete(self, name: str) -> None: ...
```

### Architektur-Regeln (verbindlich)

1. **`domain/` importiert nichts** aus `engines/`, `adapters/`, oder
   `model_registry/`. Nur `pydantic`, `typing`, stdlib.
2. **`adapters/openai_router.py` und `adapters/ollama_router.py`**
   importieren ausschließlich aus `domain/`. Sie sehen niemals
   `litert_lm`, `huggingface_hub` oder andere Engine-/Registry-Details.
3. **Wiring nur in `__main__.py`**: Hier wird `LiteRTEngine` instantiiert
   und in Router injiziert. Tests injizieren `FakeEngine`.
4. **Streaming-Konvertierung in den Adaptern**: Domain liefert
   `AsyncIterator[Token]`. OpenAI-Adapter framet als SSE (`data: …\n\n`),
   Ollama-Adapter als NDJSON (`{…}\n`). Domain weiß nichts über HTTP.
5. **App liest `config.yaml` niemals direkt** — die einzige
   Konfigurationsschnittstelle ist `os.environ` über
   `pydantic-settings`. `bashio` ist der einzige `config.yaml`-Leser im
   System.

### API-Endpoints

**OpenAI-kompatibel** (`/v1/*`):

| Method | Path | Zweck |
|---|---|---|
| POST | `/v1/chat/completions` | Chat-Inferenz, `stream: true/false` |
| POST | `/v1/completions` | Legacy-Text-Completion |
| GET | `/v1/models` | Liste verfügbarer Modelle |

**Ollama-kompatibel** (`/api/*`):

| Method | Path | Zweck |
|---|---|---|
| POST | `/api/chat` | Chat-Inferenz mit NDJSON-Stream |
| POST | `/api/generate` | Text-Inferenz mit NDJSON-Stream |
| GET | `/api/tags` | Modelle inkl. Größen |
| POST | `/api/pull` | Modell-Download starten (NDJSON-Progress) |
| POST | `/api/show` | Modell-Metadaten |
| DELETE | `/api/delete` | Modell entfernen |

**Health/Operations**:

| Method | Path | Zweck |
|---|---|---|
| GET | `/healthz` | Liveness |
| GET | `/readyz` | Readiness (Engine geladen?) |

### Bewusst nicht im MVP (YAGNI)

- Keine Embeddings-Endpoints (kommen, sobald Embedding-Modelle unterstützt werden)
- Kein Auth / API-Key (HA-internes Netz; später optional via `config.yaml`)
- Kein Request-Queue / Multi-Tenant (ein Modell-Slot, Solo-Use-Case)
- Kein Function-Calling / Tool-Use
- Kein Phi-2 (siehe Risiko in Sektion 3)
- Keine `aarch64`-Builds (Schema-vorbereitet, nicht aktiviert)
- Keine GitHub-Actions-CI (Phase 2)

## 7. CLAUDE.md-Inhalt (vollständig)

Diese Datei wird im Repo-Root angelegt:

````markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Type

Home Assistant Add-on Repository. Each top-level directory (except `.github/`,
`docs/`) is one installable add-on. The repo is consumed by Home Assistant
via the "Custom Repository" URL — `repository.yaml` is the manifest HA reads.

Distribution is currently **local-only**. Public Github + GHCR pre-built
images is the planned next phase; the layout is already compatible.

## Active Add-ons

| Add-on | Status | Purpose |
|---|---|---|
| `litert-llm-server` | in development | Local LLM inference via Google LiteRT (formerly TensorFlow Lite), exposing OpenAI- and Ollama-compatible HTTP APIs |

## Add-on Architecture Rules (apply to every add-on in this repo)

The Python application code lives in `<addon>/app/`. The HA add-on wrapper
(s6-overlay services, bashio init scripts, Dockerfile, config schema) lives
around it. These two layers are intentionally separate:

1. **`bashio` is the only translator between `config.yaml` and the Python app.**
   `rootfs/etc/cont-init.d/01-config.sh` reads add-on options via `bashio::config`
   and exports them as `LITERT_*` environment variables. The Python app reads
   **only** environment variables (via `pydantic-settings`). It must never
   parse `config.yaml` itself.

2. **s6-overlay supervises exactly one process per add-on** — the uvicorn
   server. No multi-process daemons inside the container.

3. **`/data/` is the only writable persistent path.** Downloaded models go to
   `/data/models/`. `/share/` is read-mountable for user-provided model files
   (later phase).

## Application Architecture (Ports & Adapters / Hexagonal)

Inside `<addon>/app/src/<package>/`:

```
domain/         # Pure types + Protocols. No third-party imports beyond pydantic.
engines/        # Concrete inference backends (LiteRTEngine, …).
                # Import from domain/, never the other way.
model_registry/ # Concrete model lifecycle (download, cache).
                # Import from domain/, never the other way.
adapters/       # HTTP protocol adapters (openai_router, ollama_router).
                # Import from domain/ ONLY. Never import engines/ or
                # model_registry/ directly.
__main__.py     # FastAPI app construction. The ONLY place where concrete
                # engines/registries are instantiated and injected into routers.
```

**Hard rules — these are routinely violated and must be enforced:**

- `domain/` has no imports from `engines/`, `adapters/`, or `model_registry/`.
- `adapters/openai_router.py` and `adapters/ollama_router.py` may import
  `domain/` types and call services typed against `domain/` Protocols. They
  must **never** import `litert_lm` or `huggingface_hub` directly.
- Streaming: the domain returns `AsyncIterator[Token]`. SSE framing
  (OpenAI) and NDJSON framing (Ollama) lives in the respective adapter — not
  in the engine and not in the service.

## Common Commands

All commands assume you are in `<addon-name>/app/`. The Python app uses `uv`.

```bash
# Install deps + create venv
uv sync

# Run the server locally (no HA, no Docker) — reads ENV vars
LITERT_DEFAULT_MODEL=gemma-4-e2b \
LITERT_MODELS_DIR=./.models \
LITERT_PORT=8080 \
  uv run uvicorn --factory litert_server.__main__:make_production_app --reload

# Tests
uv run pytest                       # all tests
uv run pytest tests/adapters/       # one directory
uv run pytest -k test_openai_stream # one test by name

# Type check + lint
uv run ruff check .
uv run ruff format .
uv run mypy src/
```

Building the add-on container locally (slow on first run):

```bash
# From <addon-name>/  (set BUILD_ARCH to the host's HA arch: amd64 or aarch64)
docker build \
  --build-arg BUILD_FROM=ghcr.io/hassio-addons/base-python:14.0.2 \
  --build-arg BUILD_ARCH=amd64 \
  -t local/litert-llm-server:dev .

# Run standalone for smoke tests (no HA supervisor)
docker run --rm -p 8080:8080 \
  -e LITERT_DEFAULT_MODEL=gemma-4-e2b \
  -e HF_TOKEN=hf_xxx \
  -v $(pwd)/.models:/data/models \
  local/litert-llm-server:dev
```

Installing in a local HA instance for end-to-end testing: add this repo's path
as a "Local Add-on Repository" via the HA Supervisor UI (Settings → Add-ons →
Add-on Store → ⋮ → Repositories).

## Conventions

- **Python**: 3.12+. `uv` for dep management. `ruff` for lint/format. `mypy`
  strict for new code. `pytest` + `pytest-asyncio`.
- **Versioning**: `version` in `config.yaml` is the user-visible add-on
  version. Bump on every released change, follow semver.
- **Add-on slugs**: snake_case (`litert_llm_server`). Directory names:
  kebab-case (`litert-llm-server`). The `slug` field in `config.yaml` is
  what HA stores internally.
- **Logs**: write to stdout/stderr. s6-overlay forwards to HA's add-on log
  viewer. Never write log files inside the container.

## What Lives Where

| Concern | Location |
|---|---|
| Add-on metadata, options, schema | `<addon>/config.yaml` |
| Container build | `<addon>/Dockerfile` + `<addon>/build.yaml` |
| Service supervision, init scripts | `<addon>/rootfs/etc/{services.d,cont-init.d}/` |
| Application code | `<addon>/app/src/<package>/` |
| Tests | `<addon>/app/tests/` |
| Repo manifest (HA reads this) | `repository.yaml` |
| Design specs | `docs/superpowers/specs/` |

## Out of Scope (for now)

- CI/CD via Github Actions — comes when the repo goes public.
- Multi-arch builds (aarch64, armv7) — `build.yaml` is structured for it,
  but only `amd64` is built today.
- HACS companion integration for HA — separate repo, future phase.
````

## 8. Implementierungs-Reihenfolge (für den späteren Plan)

Nur grobe Reihenfolge — die Detail-Schritte kommen aus `writing-plans`.

1. **Repo-Skelett**: `repository.yaml`, `README.md`, `CLAUDE.md` im Root.
2. **PoC-Benchmark** (kritischer Risiko-Mitigation-Step): Minimales
   Python-Skript, das Gemma-4-E2B (int4-Quantisierung, offizieller
   `litert-community/*-litert-lm`-Build) via `litert_lm.Benchmark`
   (Backend.CPU) misst. **Akzeptanzkriterium: `last_decode_tokens_per_second` ≥ 5 tok/s**
   auf der Ziel-NUC-Hardware. Bei Unterschreitung wird der Plan
   unterbrochen und die Engine-Wahl re-evaluiert (Kandidat:
   `llama-cpp-python` hinter demselben `InferenceService`-Protocol).
   Ergebnis dokumentieren in `docs/benchmarks/2026-05-XX-litert-gemma2b.md`.
3. **Domain-Schicht**: `types.py`, `inference.py`, `model_registry.py`
   (nur Protocols + Models, keine Implementierung). Tests gegen
   `FakeEngine`.
4. **Adapters** (gegen Fake-Engine entwickelbar): OpenAI-Router mit SSE,
   Ollama-Router mit NDJSON. Tests verifizieren Schema-Korrektheit gegen
   die offiziellen OpenAI/Ollama-Spec-Beispiele.
5. **`LiteRTEngine`-Implementierung**: erst jetzt, nachdem Adapter
   stehen — gegen das Protocol-Interface.
6. **`ModelRegistry`-Implementierung**: HuggingFace-Downloader + Cache.
7. **Add-on-Wrapper**: `config.yaml`, `Dockerfile`, `build.yaml`, `rootfs/`.
8. **End-to-End-Test**: Add-on in lokaler HA-Instanz installieren, HA
   „OpenAI Conversation" gegen das Add-on konfigurieren, Smoke-Test.

## 9. Erfolgs-Kriterien

- `litert-llm-server` läuft als Add-on in einer lokalen HA-Supervisor-Instanz.
- HAs eingebaute „OpenAI Conversation"-Integration spricht das Add-on
  ohne Custom-Code an und antwortet mit generiertem Text.
- Ein Ollama-Client (z. B. `ollama list`, `ollama run` via API-Override
  auf `localhost:8080`) sieht die Modelle und kann chatten.
- `/api/pull` lädt ein nicht-lokales Gemma-Modell aus HuggingFace und
  zeigt Progress.
- Alle Architektur-Regeln aus Sektion 6 sind in CI-Tests prüfbar (z. B.
  via `import-linter` oder einem Custom-Pytest-Check).

## 10. Offene Punkte für späteren Plan

- Konkrete Modell-Namensschema-Mapping: Wie wird „gemma-4-e2b" intern auf
  HuggingFace-Repo + Datei aufgelöst? Hardcoded-Tabelle im MVP, Registry
  später.
- Concurrency-Verhalten: Ein einziger Modell-Slot im RAM, zweiter
  Request während Inferenz → blockieren oder 429?
- Modell-Hot-Swap: Was passiert, wenn `/v1/chat/completions` mit anderem
  Modell aufgerufen wird als aktuell geladen? MVP: Re-Load synchron;
  Optimierung später.

## 11. Verworfene Optionen — explizit dokumentiert

- **LiteServe-Framework**: Golden Hammer für unseren Use-Case.
- **C++ MediaPipe Custom Build**: Productivity-Killer.
- **llama.cpp / Ollama als Engine**: User-Vorgabe ist LiteRT. PoC-Benchmark
  wird zeigen, ob diese Vorgabe technisch tragfähig ist; bei Misserfolg
  wird re-evaluiert.
- **`/share/models/` als primärer Modell-Pfad**: Auto-Download via
  `/api/pull` ist primärer Pfad; `/share/`-Mount bleibt vorbereitet, aber
  nicht im MVP implementiert.
- **Mehrere Add-ons im MVP**: Nur `litert-llm-server`. Weitere Add-ons
  kommen, sobald das erste produktiv ist.
