"""Engine-agnostic value types for the LLM inference domain.

This module imports nothing from `engines/`, `adapters/`, or
`model_registry/`. Only pydantic + stdlib are permitted.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FinishReason = Literal["stop", "length", "tool_calls"] | None


def new_tool_call_id() -> str:
    """Mint a client-facing tool-call id (OpenAI style)."""
    return f"call_{uuid.uuid4().hex[:24]}"


def coerce_tool_arguments(raw: Any) -> dict[str, Any]:
    """Best-effort dict from a tool-arguments payload: dict passthrough,
    JSON string parsed, anything else (or malformed JSON) → {}.

    Lives in domain/ (spec §5.3 deviation) because both the engine and the
    OpenAI adapter need it and neither may import the other."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class ToolSpec(BaseModel):
    """A function the client offers to the model (JSON-Schema parameters)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A function invocation the model asked the client to perform."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class GenerationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tokens: int = Field(ge=1, le=32768)
    temperature: float = Field(ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    # Regex the whole reply must match. Engines that support constrained
    # decoding enforce it; ignored when tools are offered (tool grammar wins).
    response_pattern: str | None = None


class Token(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    index: int = Field(ge=0)
    finish_reason: FinishReason = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def _text_xor_tool_calls(self) -> Token:
        if self.tool_calls:
            if self.text:
                raise ValueError("a token carries either text or tool_calls, not both")
            if self.finish_reason != "tool_calls":
                raise ValueError("a tool_calls token must finish with 'tool_calls'")
        return self


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
    """A single role-tagged turn in a chat conversation.

    ``tool_calls`` is set on assistant turns that requested tools;
    ``tool_name`` on ``role == "tool"`` turns whose ``content`` is the result.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_name: str | None = None
