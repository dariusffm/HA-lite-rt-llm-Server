"""Engine-agnostic value types for the LLM inference domain.

This module imports nothing from `engines/`, `adapters/`, or
`model_registry/`. Only pydantic + stdlib are permitted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FinishReason = Literal["stop", "length"] | None


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(ge=1, le=32768)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None


class Token(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    index: int = Field(ge=0)
    finish_reason: FinishReason = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    size_bytes: int = Field(ge=0)
    quantization: str
    path: Path | None = None


PullStatus = Literal["downloading", "verifying", "done", "error"]


class PullProgress(BaseModel):
    model_config = ConfigDict(frozen=True)

    bytes_done: int = Field(ge=0)
    bytes_total: int = Field(ge=0)
    status: PullStatus
    error: str | None = None


class ChatTurn(BaseModel):
    """A single role-tagged turn in a chat conversation."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
