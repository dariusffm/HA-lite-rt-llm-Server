from collections.abc import AsyncIterator

from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import ModelInfo, PullProgress


class _Concrete:
    async def list(self) -> list[ModelInfo]:
        return []

    async def get(self, name: str) -> ModelInfo:
        return ModelInfo(name=name, size_bytes=0, quantization="int4")

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        yield PullProgress(bytes_done=0, bytes_total=0, status="done")

    async def delete(self, name: str) -> None:
        return None


def test_concrete_is_structural_subtype():
    reg: ModelRegistry = _Concrete()
    assert reg is not None
