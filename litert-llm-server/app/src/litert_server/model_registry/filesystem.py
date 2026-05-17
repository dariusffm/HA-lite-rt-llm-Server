"""Local filesystem cache for `.litertlm` model files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from litert_server.domain.types import ModelInfo

MODEL_EXT = ".litertlm"


@dataclass
class FilesystemCache:
    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def scan(self) -> list[ModelInfo]:
        out: list[ModelInfo] = []
        for p in sorted(self.root.iterdir()):
            if p.is_file() and p.suffix == MODEL_EXT:
                out.append(
                    ModelInfo(
                        name=p.stem,
                        size_bytes=p.stat().st_size,
                        quantization="int4",
                        path=p,
                    )
                )
        return out

    def path_for(self, name: str) -> Path:
        return self.root / f"{name}{MODEL_EXT}"

    def delete(self, name: str) -> None:
        p = self.path_for(name)
        if p.exists():
            p.unlink()
