# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Type

Home Assistant Add-on Repository. Each top-level directory (except `.github/`,
`docs/`) is one installable add-on. The repo is consumed by Home Assistant
via the "Custom Repository" URL — `repository.yaml` is the manifest HA reads.

Distribution is via the public Github repo
`https://github.com/dariusffm/HA-lite-rt-llm-Server` (the Supervisor builds the image
on the HA host). GHCR pre-built images are the planned next phase; the layout
is already compatible.

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
  --build-arg BUILD_FROM=ghcr.io/hassio-addons/debian-base:9.3.0 \
  --build-arg BUILD_ARCH=amd64 \
  -t local/litert-llm-server:dev .

# Run standalone for smoke tests (no HA supervisor)
docker run --rm -p 8080:8080 \
  -e LITERT_DEFAULT_MODEL=gemma-4-e2b \
  -e HF_TOKEN=hf_xxx \
  -v $(pwd)/.models:/data/models \
  local/litert-llm-server:dev
```

Installing in a HA instance for end-to-end testing: add the Github URL above
as a repository via the HA Supervisor UI (Settings → Apps → App Store → ⋮ →
Repositories; older versions: Settings → Add-ons → Add-on Store).

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
