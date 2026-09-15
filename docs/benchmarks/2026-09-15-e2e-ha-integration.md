# End-to-End Test — Home Assistant with litert-llm-server

**Date:** 2026-09-15
**HA instance:** HAOS with Supervisor, aarch64 host
**Add-on version tested:** 0.1.3 (installed as 0.1.0, updated through 0.1.1 → 0.1.2 → 0.1.3 during the test)
**Repository source:** `https://github.com/dariusffm/HA-lite-rt-lm` (public, added via Settings → Apps → App Store → ⋮ → Repositories)

## Result

- Add-on installed via Git repository: **yes** (Supervisor built the image locally in ~2 min)
- Add-on started successfully: **yes** (`Uvicorn running on http://0.0.0.0:8080`)
- Model pulled via `/api/pull` (gemma-4-e2b, ~2.6 GB): **yes** — after fixes 0.1.1 and 0.1.2
- OpenAI `/v1/chat/completions` from LAN: **yes** — ~17 s cold (includes model load)
- Ollama `/api/chat` streaming from LAN: **yes** — ~4 s warm, 8 NDJSON chunks
- HA integration reached the add-on: **yes** — core **Ollama** integration, URL `http://83680c0c-litert-llm-server:8080`
- Conversation agent created (model gemma-4-e2b, Assist off): **yes** — after fix 0.1.3
- Generated response was sensible: **yes** — "Die Hauptstadt von Polen ist Warschau."

## Deviations from the plan

- The plan assumed HA's *OpenAI Conversation* integration with a custom base URL.
  The core integration has no base-URL option, so the core *Ollama* integration
  was used instead. It talks to the add-on's Ollama-compatible API.
- The plan assumed a local add-on path. A remote HAOS host cannot see the Mac
  filesystem, so the repo was pushed to Github and added by URL.

## Bugs found only in the real environment

| Version | Bug | Root cause |
|---|---|---|
| 0.1.1 | `/api/pull` → `FileNotFoundError` on `target.stat()` | `os.link()` on Linux hardlinks the HF snapshot *symlink* itself (CPython issue 37612); the relative link dangled in `/data/models`. Fixed by `os.path.realpath(src)` before linking. |
| 0.1.2 | `/api/pull` still failing after 0.1.1 | `FilesystemCache.delete` used `exists()`, which is False for a dangling symlink, so the 0.1.0 leftover survived. Fixed with `unlink(missing_ok=True)`. |
| 0.1.3 | HA Ollama "add conversation agent" → HTTP 500 `KeyError: 'model'` | `/api/tags` entries only had `name`; real Ollama emits `name` **and** `model`, and HA reads `model`. |

All three were invisible on macOS / in the unit tests (macOS `link(2)` follows
symlinks; the test fake returned a regular file; no test asserted the `model` key).
Regression tests were added for each.

## Notes

- Node-RED sanity check (plan step 5, optional) was not performed.
- `uv pip install .` in the Dockerfile ignores `uv.lock`, so the container
  resolves the newest `litert-lm-api` (0.17.0 on PyPI) while the local lock
  pins 0.11.0. Aligning the two is a follow-up.
