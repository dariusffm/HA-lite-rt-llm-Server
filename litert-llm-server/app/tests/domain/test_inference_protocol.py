from collections.abc import AsyncIterator

from litert_server.domain.inference import InferenceService
from litert_server.domain.types import GenerationParams, Token


class _Concrete:
    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        yield Token(text=prompt, index=0, finish_reason="stop")


def test_concrete_is_structural_subtype():
    svc: InferenceService = _Concrete()
    assert svc is not None
