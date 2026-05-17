"""HuggingFace-backed `ModelRegistry` implementation.

MVP strategy: a hardcoded catalog mapping public model names to HF repo +
filename. Half the catalog is public (`litert-community/*`), half is
gated (`google/*-litert-lm`, needs ``hf_token``).
"""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from litert_server.domain.types import ModelInfo, PullProgress
from litert_server.model_registry.filesystem import FilesystemCache


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    repo_id: str
    filename: str
    quantization: str
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


@dataclass
class HuggingFaceRegistry:
    cache: FilesystemCache
    hf_token: str | None = None  # injected by __main__ from Settings.hf_token

    async def list(self) -> list[ModelInfo]:
        return self.cache.scan()

    async def get(self, name: str) -> ModelInfo:
        for m in self.cache.scan():
            if m.name == name:
                return m
        raise KeyError(name)

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
        await asyncio.to_thread(shutil.copyfile, tmp_path, target)

        size = target.stat().st_size
        yield PullProgress(bytes_done=size, bytes_total=size, status="done")

    async def delete(self, name: str) -> None:
        self.cache.delete(name)
