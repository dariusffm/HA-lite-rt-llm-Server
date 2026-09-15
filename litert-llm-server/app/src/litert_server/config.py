"""Typed settings read exclusively from environment variables.

bashio is the only translator between `config.yaml` and these env vars
(see `rootfs/etc/cont-init.d/01-config.sh`). The application MUST NOT
parse `config.yaml` itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal[
    "trace", "debug", "info", "notice", "warning", "error", "fatal"
]


class Settings(BaseSettings):
    """Add-on settings exposed via the ``LITERT_*`` env-var prefix, plus a
    special ``hf_token`` field that reads the **un-prefixed** ``HF_TOKEN``
    env var (the standard variable that ``huggingface_hub`` recognizes).
    """

    model_config = SettingsConfigDict(env_prefix="LITERT_", extra="ignore")

    log_level: LogLevel = "info"
    default_model: str = "gemma-4-e2b"
    max_tokens: int = 1024
    temperature: float = 0.7
    models_dir: Path = Path("/data/models")
    port: int = 8080
    preload_models: list[str] = []
    tool_calling: bool = True
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    @field_validator("preload_models", mode="before")
    @classmethod
    def _parse_preload(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            return json.loads(v)
        return v

    @field_validator("hf_token", mode="before")
    @classmethod
    def _empty_token_is_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v
