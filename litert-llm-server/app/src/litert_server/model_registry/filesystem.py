"""Local filesystem cache for `.litertlm` model files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from litert_server.domain.model_names import model_path
from litert_server.domain.types import ModelInfo

MODEL_EXT = ".litertlm"


@dataclass
class FilesystemCache:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def scan(self) -> list[ModelInfo]:
        """List cached models. Quantization is reported as ``"unknown"``;
        callers (registries) enrich it via their catalog.
        """
        out: list[ModelInfo] = []
        for p in sorted(self.root.iterdir()):
            if p.is_file() and p.suffix == MODEL_EXT:
                out.append(
                    ModelInfo(
                        name=p.stem,
                        size_bytes=p.stat().st_size,
                        quantization="unknown",
                        path=p,
                    )
                )
        return out

    def path_for(self, name: str) -> Path:
        return model_path(self.root, name, MODEL_EXT)

    def delete(self, name: str) -> None:
        # missing_ok also removes dangling symlinks, which ``exists()`` hides.
        self.path_for(name).unlink(missing_ok=True)
