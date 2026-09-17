"""Fake `ModelRegistry` for adapter and wiring tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from litert_server.domain.types import ModelInfo, PullProgress


@dataclass
class FakeRegistry:
    models: dict[str, ModelInfo] = field(
        default_factory=lambda: {
            "gemma-4-e2b": ModelInfo(
                name="gemma-4-e2b", size_bytes=1_500_000_000, quantization="int4"
            ),
        }
    )
    pull_chunks: int = 3

    async def list(self) -> list[ModelInfo]:
        return list(self.models.values())

    async def get(self, name: str) -> ModelInfo:
        if name not in self.models:
            raise KeyError(name)
        return self.models[name]

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        total = 1_000
        for i in range(1, self.pull_chunks + 1):
            yield PullProgress(
                bytes_done=int(total * i / self.pull_chunks),
                bytes_total=total,
                status="downloading" if i < self.pull_chunks else "done",
            )
        if name not in self.models:
            self.models[name] = ModelInfo(name=name, size_bytes=total, quantization="int4")

    async def delete(self, name: str) -> None:
        self.models.pop(name, None)
