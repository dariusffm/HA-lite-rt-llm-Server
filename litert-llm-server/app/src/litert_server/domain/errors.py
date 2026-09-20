"""Domain-level failures the adapters translate into HTTP status codes.

They live here so `engines/` can raise them and `adapters/` can catch them
by name without either importing the other.
"""

from __future__ import annotations


class InferenceError(Exception):
    """Base class for failures the caller can act on."""


class EngineBusyError(InferenceError):
    """The engine is occupied and the caller waited long enough.

    The engine serves one generation at a time. Adapters answer 503 with a
    `Retry-After`: the request is fine, the moment is not.
    """


class EngineUnavailableError(InferenceError):
    """The engine can no longer be used safely.

    Raised when a generation's native work did not confirm it had finished:
    a thread inside the runtime is unaccounted for, so handing the engine to
    the next request could close resources it is still using. Not
    recoverable in-process — the service reports itself unhealthy and exits
    so the supervisor restarts it.
    """
