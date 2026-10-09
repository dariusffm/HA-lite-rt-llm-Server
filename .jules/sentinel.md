## 2026-03-30 - Adapter Error Message Information Leakage
**Vulnerability:** Unhandled 500 exceptions in OpenAI and Ollama HTTP adapters returned `str(exc)` in JSON and stream error frames, exposing internal path and exception details to clients.
**Learning:** Domain errors intended for clients (e.g. `InvalidModelNameError`, `EngineBusyError`) should be explicitly caught with client-friendly status codes/messages, while general unexpected exceptions must return generic "Internal server error" messages while logging details server-side.
**Prevention:** Ensure all FastAPI router error handlers distinguish between client-safe domain exceptions and generic unhandled exceptions when constructing error JSON and SSE/NDJSON frames.
