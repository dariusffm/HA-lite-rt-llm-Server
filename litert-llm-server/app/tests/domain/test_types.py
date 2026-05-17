import pytest
from pydantic import ValidationError

from litert_server.domain.types import (
    GenerationParams,
    ModelInfo,
    PullProgress,
    Token,
)


def test_token_carries_text_and_index():
    t = Token(text="hello", index=0)
    assert t.text == "hello"
    assert t.index == 0
    assert t.finish_reason is None


def test_token_finish_reason_must_be_known():
    Token(text="x", index=1, finish_reason="stop")
    Token(text="x", index=1, finish_reason="length")
    Token(text="x", index=1, finish_reason=None)
    with pytest.raises(ValidationError):
        Token(text="x", index=1, finish_reason="invented")  # type: ignore[arg-type]


def test_generation_params_clamps_via_validation():
    p = GenerationParams(max_tokens=256, temperature=0.7)
    assert p.top_p is None
    assert p.stop is None


def test_model_info_path_is_optional():
    m = ModelInfo(name="gemma-4-e2b", size_bytes=1234, quantization="int4")
    assert m.path is None


def test_pull_progress_error_only_with_error_status():
    PullProgress(bytes_done=10, bytes_total=100, status="downloading")
    PullProgress(bytes_done=100, bytes_total=100, status="done")
    PullProgress(bytes_done=0, bytes_total=0, status="error", error="boom")
