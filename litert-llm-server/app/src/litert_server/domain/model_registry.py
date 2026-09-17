"""`ModelRegistry` Protocol — model lifecycle (list/get/pull/delete)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from litert_server.domain.types import ModelInfo, PullProgress


@runtime_checkable
class ModelRegistry(Protocol):
    """A model lifecycle manager.

    Implementations live in `model_registry/`. Adapters must type their
    dependency against this Protocol — never against a concrete class.
    """

    async def list(self) -> list[ModelInfo]: ...

    async def get(self, name: str) -> ModelInfo: ...

    def pull(self, name: str) -> AsyncIterator[PullProgress]: ...

    async def delete(self, name: str) -> None: ...
