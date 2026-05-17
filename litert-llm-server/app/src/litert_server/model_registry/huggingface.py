"""HuggingFace-backed `ModelRegistry` implementation.

MVP strategy: a hardcoded catalog mapping public model names to HF repo +
filename. Half the catalog is public (`litert-community/*`), half is
gated (`google/*-litert-lm`, needs ``hf_token``).
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from litert_server.domain.types import ModelInfo, PullProgress
from litert_server.model_registry.filesystem import FilesystemCache


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    repo_id: str
    filename: str
    quantization: Literal["int4"]
    gated: bool


MODEL_CATALOG: dict[str, CatalogEntry] = {
    # Public — no HF token required.
    "gemma-4-e2b": CatalogEntry(
        name="gemma-4-e2b",
        repo_id="litert-community/gemma-4-E2B-it-litert-lm",
        filename="gemma-4-E2B-it.litertlm",
        quantization="int4",
        gated=False,
    ),
    "gemma-4-e4b": CatalogEntry(
        name="gemma-4-e4b",
        repo_id="litert-community/gemma-4-E4B-it-litert-lm",
        filename="gemma-4-E4B-it.litertlm",
        quantization="int4",
        gated=False,
    ),
    # Gated — require HF token + accepted Gemma license on the model page.
    "gemma-3n-e2b": CatalogEntry(
        name="gemma-3n-e2b",
        repo_id="google/gemma-3n-E2B-it-litert-lm",
        filename="gemma-3n-E2B-it-int4.litertlm",
        quantization="int4",
        gated=True,
    ),
    "gemma-3n-e4b": CatalogEntry(
        name="gemma-3n-e4b",
        repo_id="google/gemma-3n-E4B-it-litert-lm",
        filename="gemma-3n-E4B-it-int4.litertlm",
        quantization="int4",
        gated=True,
    ),
}


def _link_or_copy(src: str, dst: Path) -> None:
    """Materialize ``dst`` from ``src``. Tries a hardlink first (zero-copy
    on the same filesystem) and falls back to a full copy across mounts.
    """
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


@dataclass
class HuggingFaceRegistry:
    cache: FilesystemCache
    hf_token: str | None = None  # injected by __main__ from Settings.hf_token

    def __post_init__(self) -> None:
        # Empty string from bashio's `config.yaml` should behave like 'no token'.
        if self.hf_token is not None and not self.hf_token.strip():
            self.hf_token = None

    def _enrich(self, info: ModelInfo) -> ModelInfo:
        entry = MODEL_CATALOG.get(info.name)
        if entry is None:
            return info
        return info.model_copy(update={"quantization": entry.quantization})

    async def list(self) -> list[ModelInfo]:
        return [self._enrich(m) for m in self.cache.scan()]

    async def get(self, name: str) -> ModelInfo:
        path = self.cache.path_for(name)
        if not path.is_file():
            raise KeyError(name)
        stat = path.stat()
        entry = MODEL_CATALOG.get(name)
        return ModelInfo(
            name=name,
            size_bytes=stat.st_size,
            quantization=entry.quantization if entry else "unknown",
            path=path,
        )

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        if name not in MODEL_CATALOG:
            yield PullProgress(
                bytes_done=0,
                bytes_total=0,
                status="error",
                error=f"unknown model '{name}'",
            )
            return

        entry = MODEL_CATALOG[name]
        yield PullProgress(bytes_done=0, bytes_total=0, status="downloading")

        def _download() -> str:
            return hf_hub_download(
                repo_id=entry.repo_id,
                filename=entry.filename,
                cache_dir=str(self.cache.root / ".hf_cache"),
                token=self.hf_token,
            )

        try:
            tmp_path = await asyncio.to_thread(_download)
        except Exception as exc:
            yield PullProgress(
                bytes_done=0, bytes_total=0, status="error", error=str(exc)
            )
            return

        target = self.cache.path_for(name)
        # Replace any prior file (delete-or-noop, then link/copy).
        await asyncio.to_thread(self.cache.delete, name)
        await asyncio.to_thread(_link_or_copy, tmp_path, target)

        size = target.stat().st_size
        yield PullProgress(bytes_done=size, bytes_total=size, status="done")

    async def delete(self, name: str) -> None:
        self.cache.delete(name)
