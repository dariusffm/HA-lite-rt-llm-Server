# Changelog

## 0.1.2 — 2026-09-15

- Fix `/api/pull` still failing after 0.1.1 when a dangling model symlink
  (left behind by 0.1.0) exists: `FilesystemCache.delete` now removes
  dangling symlinks before the pulled file is materialized.

## 0.1.1 — 2026-09-15

- Fix `/api/pull` failing with `FileNotFoundError` on Linux: the HuggingFace
  snapshot symlink was hardlinked instead of the blob it points to
  (CPython issue 37612). The source path is now resolved before linking.
- Repository metadata points to the public Github repo; MIT license added.

## 0.1.0 — 2026-05-17

- Initial release.
- OpenAI- and Ollama-compatible APIs.
- Gemma 4 E2B/E4B (public) and Gemma 3n E2B/E4B (gated) via HuggingFace catalog.
- Auto-download via `/api/pull`.
- Model-native chat templating via litert_lm Conversation API.
- 32k context window.
- amd64 + aarch64 builds.
