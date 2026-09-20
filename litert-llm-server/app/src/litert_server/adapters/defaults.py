"""Fallback values the add-on options supply to both HTTP adapters.

`config.yaml` has always exposed `default_model`, `max_tokens` and
`temperature`, but the adapters carried their own literals, so the options
had no effect. They are grouped here rather than passed as three more
keyword arguments: they are one concept — "what to use when the client
says nothing" — and the next option would otherwise widen both builders
again.

`tools_enabled` deliberately stays a separate parameter: it is a feature
switch, not a value a request can override.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationDefaults:
    """Defaults applied per request when the client omits a field.

    The values mirror the literals the adapters used before, so a router
    built without explicit defaults behaves exactly as it did.
    """

    model: str = "gemma-4-e2b"
    max_tokens: int = 512
    temperature: float = 0.7
