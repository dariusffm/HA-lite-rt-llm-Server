"""Model-name validation.

Model names reach the app from untrusted HTTP bodies (``model`` in both
adapters, ``name`` in Ollama's pull/show/delete) and are turned into
filesystem paths by ``FilesystemCache.path_for`` and
``LiteRTEngine._model_file``. Without a check, ``../..`` escapes the
models directory — ``DELETE /api/delete`` would unlink files outside it.

Validation lives here so both call sites share one rule, and the path
builders additionally confirm the resolved path stays inside their root
(a symlink can satisfy the pattern and still point elsewhere).
"""

from __future__ import annotations

import re
from pathlib import Path

#: Catalog names are ASCII, start alphanumeric and stay short (``gemma-4-e2b``).
#: Notably excluded: ``/``, ``\``, ``..`` as a whole name, and leading dots.
_MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class InvalidModelNameError(ValueError):
    """A model name is unusable as a filename. Adapters map this to HTTP 400."""


def validate_model_name(name: str) -> str:
    """Return ``name`` unchanged, or raise ``InvalidModelNameError``."""
    if not _MODEL_NAME.match(name) or ".." in name:
        raise InvalidModelNameError(f"invalid model name: {name!r}")
    return name


def model_path(root: Path, name: str, suffix: str) -> Path:
    """Build ``root/<name><suffix>`` for a validated name, confirming the
    result stays inside ``root`` even if a symlink points away.
    """
    validate_model_name(name)
    path = root / f"{name}{suffix}"
    try:
        inside = path.resolve().is_relative_to(root.resolve())
    except OSError as exc:  # unresolvable path: treat as outside
        raise InvalidModelNameError(f"invalid model name: {name!r}") from exc
    if not inside:
        raise InvalidModelNameError(f"invalid model name: {name!r}")
    return path
