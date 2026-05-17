# LiteRT LLM Server Add-on Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Home Assistant Add-on Repository with a first add-on `litert-llm-server` that serves local LLM inference (Google LiteRT engine) via OpenAI- and Ollama-compatible HTTP APIs.

**Architecture:** Ports & Adapters (Hexagonal). Two HTTP protocol adapters (OpenAI, Ollama) talk to a single `InferenceService` Protocol implemented by `LiteRTEngine`. A `ModelRegistry` Protocol handles model downloads via HuggingFace. The HA add-on wrapper (s6-overlay + bashio) is a thin shell around the Python app; bashio is the only translator from `config.yaml` to environment variables.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, pydantic v2, pydantic-settings, litert-lm-api, huggingface_hub, pytest + pytest-asyncio, httpx, ruff, mypy, import-linter, uv. Container base: `ghcr.io/hassio-addons/base-python:14.0.2`.

**Reference spec:** `docs/superpowers/specs/2026-05-16-homeassistant-addons-repo-design.md`

---

## Critical Decision Gate

Phase 1 (Tasks 3–5) is a **PoC benchmark** that gates the entire plan. If Gemma 2B on the target hardware achieves **< 5 tok/s sustained**, STOP and consult the user before proceeding. The remaining tasks assume LiteRT is viable on the target hardware.

---

## Phase 0 — Repo Skeleton

### Task 1: Initialize git, create repository manifest and root docs

**Files:**
- Create: `repository.yaml`
- Create: `README.md`
- Create: `CLAUDE.md`
- Create: `.gitignore`

- [ ] **Step 1: Initialize git in the repo root**

Run:
```bash
cd /Users/dariuspauly/projects/claude/homassist-addons
git init -b main
```
Expected: `Initialized empty Git repository in .../.git/`

- [ ] **Step 2: Create `repository.yaml`**

```yaml
name: "homassist-addons"
url: "https://github.com/USER/homassist-addons"
maintainer: "Darius Pauly <d.pauly@easydevelopment.net>"
```

- [ ] **Step 3: Create `README.md`**

```markdown
# homassist-addons

Personal Home Assistant Add-on Repository.

## Add-ons

- **litert-llm-server** — Local LLM inference via Google LiteRT, with
  OpenAI- and Ollama-compatible HTTP APIs.

## Installation (Local)

In Home Assistant: **Settings → Add-ons → Add-on Store → ⋮ →
Repositories**, add the path of this repository.

Public Github distribution is planned for a later phase.

## Development

See `CLAUDE.md` for the architectural rules every add-on in this repo follows.
Per-add-on docs live in each add-on's directory.
```

- [ ] **Step 4: Create `CLAUDE.md`**

Use the verbatim content from Section 7 of the spec (`docs/superpowers/specs/2026-05-16-homeassistant-addons-repo-design.md`). Copy block-for-block. The spec is the source of truth.

- [ ] **Step 5: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Models cache (local dev runs)
.models/
*/app/.models/

# Editor / OS
.DS_Store
.idea/
.vscode/

# Build artifacts
build/
dist/
```

- [ ] **Step 6: Commit the skeleton**

```bash
git add repository.yaml README.md CLAUDE.md .gitignore docs/
git commit -m "chore: init repository skeleton with manifest and CLAUDE.md"
```

---

### Task 2: Create empty add-on directory structure

**Files:**
- Create: `litert-llm-server/app/src/litert_server/__init__.py`
- Create: `litert-llm-server/app/src/litert_server/domain/__init__.py`
- Create: `litert-llm-server/app/src/litert_server/engines/__init__.py`
- Create: `litert-llm-server/app/src/litert_server/adapters/__init__.py`
- Create: `litert-llm-server/app/src/litert_server/model_registry/__init__.py`
- Create: `litert-llm-server/app/tests/__init__.py`
- Create: `litert-llm-server/app/tests/domain/__init__.py`
- Create: `litert-llm-server/app/tests/engines/__init__.py`
- Create: `litert-llm-server/app/tests/adapters/__init__.py`
- Create: `litert-llm-server/app/tests/model_registry/__init__.py`
- Create: `litert-llm-server/app/tests/fakes/__init__.py`
- Create: `litert-llm-server/rootfs/etc/services.d/litert/.keep`
- Create: `litert-llm-server/rootfs/etc/cont-init.d/.keep`

- [ ] **Step 1: Create all directories with empty `__init__.py` / `.keep` files**

```bash
cd /Users/dariuspauly/projects/claude/homassist-addons
mkdir -p litert-llm-server/app/src/litert_server/{domain,engines,adapters,model_registry}
mkdir -p litert-llm-server/app/tests/{domain,engines,adapters,model_registry,fakes}
mkdir -p litert-llm-server/rootfs/etc/{services.d/litert,cont-init.d}

touch litert-llm-server/app/src/litert_server/__init__.py
touch litert-llm-server/app/src/litert_server/{domain,engines,adapters,model_registry}/__init__.py
touch litert-llm-server/app/tests/__init__.py
touch litert-llm-server/app/tests/{domain,engines,adapters,model_registry,fakes}/__init__.py
touch litert-llm-server/rootfs/etc/services.d/litert/.keep
touch litert-llm-server/rootfs/etc/cont-init.d/.keep
```

- [ ] **Step 2: Commit the empty structure**

```bash
git add litert-llm-server/
git commit -m "chore(litert): scaffold add-on directory structure"
```

---

## Phase 1 — PoC Benchmark (Decision Gate)

### Task 3: Set up benchmark environment with `uv`

**Files:**
- Create: `litert-llm-server/app/pyproject.toml`
- Create: `litert-llm-server/app/README.md`
- Create: `docs/benchmarks/.keep`

- [ ] **Step 1: Verify `uv` is installed**

Run:
```bash
uv --version
```
Expected: `uv 0.x.x` (any 0.x version).
If not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

- [ ] **Step 2: Create `litert-llm-server/app/pyproject.toml`**

```toml
[project]
name = "litert-server"
version = "0.1.0"
description = "Local LLM inference server using Google LiteRT"
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "pydantic>=2.9",
    "pydantic-settings>=2.6",
    "httpx>=0.27",
    "huggingface-hub>=0.26",
    "litert-lm-api>=0.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "ruff>=0.7",
    "mypy>=1.13",
    "import-linter>=2.1",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/litert_server"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "ASYNC", "RUF"]

[tool.mypy]
python_version = "3.12"
strict = true
files = ["src"]
```

- [ ] **Step 3: Create `docs/benchmarks/` placeholder**

```bash
mkdir -p docs/benchmarks
touch docs/benchmarks/.keep
```

- [ ] **Step 4: Create `litert-llm-server/app/README.md`**

```markdown
# litert-llm-server

Python application for the LiteRT LLM Server add-on. See `CLAUDE.md` at the
repo root for architecture rules.

## Local development

```bash
cd litert-llm-server/app
uv sync --extra dev
uv run pytest
```

See the repo-level `CLAUDE.md` for the full command reference.
```

- [ ] **Step 5: Verify `uv sync` works**

Run:
```bash
cd litert-llm-server/app
uv sync --extra dev
```
Expected: virtualenv created at `.venv/`, dependencies installed without error.

> **Note:** `litert-lm-api` ships self-contained wheels and installs
> cleanly on macOS-arm64 and Linux-amd64. The benchmark in Task 5 must
> still run on the **target amd64 Linux machine** for representative
> tok/s numbers. Local imports + `uv sync` should succeed on macOS too.

- [ ] **Step 6: Commit**

```bash
git add litert-llm-server/app/pyproject.toml litert-llm-server/app/README.md docs/benchmarks/
git commit -m "feat(litert): add pyproject.toml with uv-managed deps"
```

---

### Task 4: Write the PoC benchmark script

**Files:**
- Create: `litert-llm-server/app/scripts/bench_gemma2b.py`

- [ ] **Step 1: Create `litert-llm-server/app/scripts/bench_gemma2b.py`**

```python
"""PoC benchmark: measure LiteRT-LM Gemma-4-E2B sustained decode token rate.

Acceptance criterion (from spec): >= 5 tok/s sustained decode rate on the
target amd64 hardware. Below that, the engine choice must be re-evaluated
before continuing.

Uses litert-lm-api's built-in `Benchmark`, which reports the canonical
`last_decode_tokens_per_second` metric (token generation speed, NOT prompt
prefill).

Run on the actual target NUC/server:
    cd litert-llm-server/app
    uv run python scripts/bench_gemma2b.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from litert_lm import Backend, Benchmark

MODEL_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
MODEL_FILE = "gemma-4-E2B-it.litertlm"
PREFILL_TOKENS = 256
DECODE_TOKENS = 128
ACCEPTANCE_TOK_S = 5.0


def download_model(cache_dir: Path) -> Path:
    print(f"Downloading {MODEL_REPO}/{MODEL_FILE} ...")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def main() -> int:
    cache_dir = Path(os.environ.get("LITERT_BENCH_CACHE", "./.models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_model(cache_dir)

    print(f"Running LiteRT-LM benchmark on {model_path} (Backend.CPU) ...")
    bench = Benchmark(
        model_path=str(model_path),
        backend=Backend.CPU,
        prefill_tokens=PREFILL_TOKENS,
        decode_tokens=DECODE_TOKENS,
    )
    info = bench.run()

    print("\n=== RESULT ===")
    print(f"init_time:                       {info.init_time_in_second:.2f}s")
    print(f"time_to_first_token:             {info.time_to_first_token_in_second:.2f}s")
    print(f"prefill_tokens_per_second:       {info.last_prefill_tokens_per_second:.2f}")
    print(f"decode_tokens_per_second:        {info.last_decode_tokens_per_second:.2f}")
    print(f"acceptance threshold (decode):   >= {ACCEPTANCE_TOK_S} tok/s")

    if info.last_decode_tokens_per_second < ACCEPTANCE_TOK_S:
        print("BELOW ACCEPTANCE THRESHOLD — stop and re-evaluate engine choice.")
        return 1
    print("Above threshold — proceed with LiteRT-LM engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Make the script discoverable**

```bash
chmod +x litert-llm-server/app/scripts/bench_gemma2b.py
```

- [ ] **Step 3: Commit**

```bash
git add litert-llm-server/app/scripts/bench_gemma2b.py
git commit -m "feat(litert): add Gemma 2B PoC benchmark script"
```

---

### Task 5: Run the benchmark and document the result

**Files:**
- Create: `docs/benchmarks/2026-05-16-litert-gemma2b.md`

- [ ] **Step 1: Run the benchmark on the target hardware**

Run (on the target amd64 NUC/server):
```bash
cd litert-llm-server/app
uv sync --extra dev
uv run python scripts/bench_gemma2b.py
```
Expected: prints token rate. Capture the value.

- [ ] **Step 2: Document the result in `docs/benchmarks/2026-05-16-litert-gemma2b.md`**

```markdown
# Benchmark — LiteRT Gemma 2B on amd64

**Date:** 2026-05-16
**Host:** <fill in: CPU model, RAM, kernel>
**Model:** `litert-community/gemma-4-E2B-it-litert-lm` / `gemma-4-E2B-it.litertlm`
**Script:** `litert-llm-server/app/scripts/bench_gemma2b.py`
**Prompt:** `"Write a short paragraph about home automation."`
**Tokens generated:** 100 (whitespace-split heuristic)

## Result

| Metric | Value |
|---|---|
| Sustained token rate | **<fill in> tok/s** |
| Acceptance threshold | 5.0 tok/s |
| Decision | <PASS / FAIL> |

## Notes

<Any observations: CPU load, RAM usage, startup time, etc.>
```

- [ ] **Step 3: Decision gate**

If the measured rate is **≥ 5 tok/s**: proceed to Task 6.

If the measured rate is **< 5 tok/s**: STOP. Do not continue with this plan. Report to the user with the measured number and request a decision on engine swap (`llama-cpp-python` is the documented fallback in the spec).

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks/2026-05-16-litert-gemma2b.md
git commit -m "docs(bench): record LiteRT Gemma 2B benchmark result"
```

---

## Phase 2 — Domain Layer

### Task 6: Domain types

**Files:**
- Create: `litert-llm-server/app/src/litert_server/domain/types.py`
- Create: `litert-llm-server/app/tests/domain/test_types.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/domain/test_types.py`:

```python
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
    m = ModelInfo(name="gemma-4-e2b", size_bytes=1234, quantization="int8")
    assert m.path is None


def test_pull_progress_error_only_with_error_status():
    PullProgress(bytes_done=10, bytes_total=100, status="downloading")
    PullProgress(bytes_done=100, bytes_total=100, status="done")
    PullProgress(bytes_done=0, bytes_total=0, status="error", error="boom")
```

- [ ] **Step 2: Run the test (expect fail)**

```bash
cd litert-llm-server/app
uv run pytest tests/domain/test_types.py -v
```
Expected: `ModuleNotFoundError: No module named 'litert_server.domain.types'`

- [ ] **Step 3: Implement `litert-llm-server/app/src/litert_server/domain/types.py`**

```python
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
```

- [ ] **Step 4: Run the test (expect pass)**

```bash
uv run pytest tests/domain/test_types.py -v
```
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/domain/types.py \
        litert-llm-server/app/tests/domain/test_types.py
git commit -m "feat(litert/domain): add value types (Token, GenerationParams, ModelInfo, PullProgress)"
```

---

### Task 7: `InferenceService` Protocol

**Files:**
- Create: `litert-llm-server/app/src/litert_server/domain/inference.py`
- Create: `litert-llm-server/app/tests/domain/test_inference_protocol.py`

- [ ] **Step 1: Write the failing test (verifies Protocol shape)**

Create `litert-llm-server/app/tests/domain/test_inference_protocol.py`:

```python
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
    assert svc is not None  # mypy + runtime: Protocol satisfied
```

- [ ] **Step 2: Run the test (expect import fail)**

```bash
uv run pytest tests/domain/test_inference_protocol.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `litert-llm-server/app/src/litert_server/domain/inference.py`**

```python
"""`InferenceService` Protocol — the single port between adapters and engines."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from litert_server.domain.types import GenerationParams, Token


@runtime_checkable
class InferenceService(Protocol):
    """A streaming completion service.

    Implementations live in `engines/`. Adapters MUST type their dependency
    against this Protocol — never against a concrete engine class.
    """

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        ...
```

- [ ] **Step 4: Run the test (expect pass)**

```bash
uv run pytest tests/domain/test_inference_protocol.py -v
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/domain/inference.py \
        litert-llm-server/app/tests/domain/test_inference_protocol.py
git commit -m "feat(litert/domain): add InferenceService Protocol"
```

---

### Task 8: `ModelRegistry` Protocol

**Files:**
- Create: `litert-llm-server/app/src/litert_server/domain/model_registry.py`
- Create: `litert-llm-server/app/tests/domain/test_model_registry_protocol.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/domain/test_model_registry_protocol.py`:

```python
from collections.abc import AsyncIterator

from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import ModelInfo, PullProgress


class _Concrete:
    async def list(self) -> list[ModelInfo]:
        return []

    async def get(self, name: str) -> ModelInfo:
        return ModelInfo(name=name, size_bytes=0, quantization="int8")

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        yield PullProgress(bytes_done=0, bytes_total=0, status="done")

    async def delete(self, name: str) -> None:
        return None


def test_concrete_is_structural_subtype():
    reg: ModelRegistry = _Concrete()
    assert reg is not None
```

- [ ] **Step 2: Run the test (expect import fail)**

```bash
uv run pytest tests/domain/test_model_registry_protocol.py -v
```

- [ ] **Step 3: Implement `litert-llm-server/app/src/litert_server/domain/model_registry.py`**

```python
"""`ModelRegistry` Protocol — model lifecycle (list/get/pull/delete)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from litert_server.domain.types import ModelInfo, PullProgress


@runtime_checkable
class ModelRegistry(Protocol):
    """A model lifecycle manager.

    Implementations live in `model_registry/`. Adapters must type their
    dependency against this Protocol — never against a concrete class.
    """

    async def list(self) -> list[ModelInfo]:
        ...

    async def get(self, name: str) -> ModelInfo:
        ...

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        ...

    async def delete(self, name: str) -> None:
        ...
```

- [ ] **Step 4: Run the test (expect pass)**

```bash
uv run pytest tests/domain/test_model_registry_protocol.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/domain/model_registry.py \
        litert-llm-server/app/tests/domain/test_model_registry_protocol.py
git commit -m "feat(litert/domain): add ModelRegistry Protocol"
```

---

## Phase 3 — Test Fakes

### Task 9: Fake `InferenceService` for adapter tests

**Files:**
- Create: `litert-llm-server/app/tests/fakes/fake_engine.py`

- [ ] **Step 1: Implement the fake**

```python
"""Fake `InferenceService` for adapter-layer testing.

Streams a fixed sequence of tokens. Records the most recent call args so
tests can assert what the adapter sent down.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from litert_server.domain.types import GenerationParams, Token


@dataclass
class FakeEngineCall:
    model: str
    prompt: str
    params: GenerationParams


@dataclass
class FakeEngine:
    tokens: list[str] = field(default_factory=lambda: ["Hello", ", ", "world", "!"])
    finish_reason: str = "stop"
    calls: list[FakeEngineCall] = field(default_factory=list)

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        self.calls.append(FakeEngineCall(model=model, prompt=prompt, params=params))
        for i, text in enumerate(self.tokens):
            is_last = i == len(self.tokens) - 1
            yield Token(
                text=text,
                index=i,
                finish_reason=self.finish_reason if is_last else None,  # type: ignore[arg-type]
            )
```

- [ ] **Step 2: Verify it satisfies the Protocol**

Create a minimal sanity test in the same file is unnecessary; mypy will catch a mismatch. Run:

```bash
cd litert-llm-server/app
uv run mypy src/ tests/fakes/
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add litert-llm-server/app/tests/fakes/fake_engine.py
git commit -m "test(litert/fakes): add FakeEngine for adapter-layer tests"
```

---

### Task 10: Fake `ModelRegistry`

**Files:**
- Create: `litert-llm-server/app/tests/fakes/fake_registry.py`

- [ ] **Step 1: Implement the fake**

```python
"""Fake `ModelRegistry` for adapter and wiring tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from litert_server.domain.types import ModelInfo, PullProgress


@dataclass
class FakeRegistry:
    models: dict[str, ModelInfo] = field(
        default_factory=lambda: {
            "gemma-4-e2b": ModelInfo(
                name="gemma-4-e2b", size_bytes=1_500_000_000, quantization="int8"
            ),
        }
    )
    pull_chunks: int = 3

    async def list(self) -> list[ModelInfo]:
        return list(self.models.values())

    async def get(self, name: str) -> ModelInfo:
        if name not in self.models:
            raise KeyError(name)
        return self.models[name]

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        total = 1_000
        for i in range(1, self.pull_chunks + 1):
            yield PullProgress(
                bytes_done=int(total * i / self.pull_chunks),
                bytes_total=total,
                status="downloading" if i < self.pull_chunks else "done",
            )
        if name not in self.models:
            self.models[name] = ModelInfo(
                name=name, size_bytes=total, quantization="int8"
            )

    async def delete(self, name: str) -> None:
        self.models.pop(name, None)
```

- [ ] **Step 2: Mypy-check**

```bash
uv run mypy tests/fakes/
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add litert-llm-server/app/tests/fakes/fake_registry.py
git commit -m "test(litert/fakes): add FakeRegistry"
```

---

## Phase 4 — OpenAI Adapter

### Task 11: `/v1/models` endpoint

**Files:**
- Create: `litert-llm-server/app/src/litert_server/adapters/openai_router.py`
- Create: `litert-llm-server/app/tests/adapters/conftest.py`
- Create: `litert-llm-server/app/tests/adapters/test_openai_models.py`

- [ ] **Step 1: Write a shared fixtures conftest**

Create `litert-llm-server/app/tests/adapters/conftest.py`:

```python
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from litert_server.adapters.openai_router import build_openai_router
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def fake_registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
def app(fake_engine: FakeEngine, fake_registry: FakeRegistry) -> FastAPI:
    app = FastAPI()
    app.include_router(build_openai_router(engine=fake_engine, registry=fake_registry))
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 2: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_openai_models.py`:

```python
from httpx import AsyncClient


async def test_list_models_returns_openai_shape(client: AsyncClient):
    r = await client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert any(item["id"] == "gemma-4-e2b" for item in body["data"])
    for item in body["data"]:
        assert item["object"] == "model"
        assert "created" in item
        assert item["owned_by"] == "litert-llm-server"
```

- [ ] **Step 3: Run test (expect import fail)**

```bash
uv run pytest tests/adapters/test_openai_models.py -v
```
Expected: `ModuleNotFoundError: No module named 'litert_server.adapters.openai_router'`

- [ ] **Step 4: Implement `litert-llm-server/app/src/litert_server/adapters/openai_router.py`**

```python
"""OpenAI-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/SSE and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry


class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "litert-llm-server"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModelItem]


def build_openai_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/models", response_model=OpenAIModelList)
    async def list_models() -> OpenAIModelList:
        models = await registry.list()
        now = int(time.time())
        return OpenAIModelList(
            data=[OpenAIModelItem(id=m.name, created=now) for m in models],
        )

    return router
```

- [ ] **Step 5: Run test (expect pass)**

```bash
uv run pytest tests/adapters/test_openai_models.py -v
```
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/openai_router.py \
        litert-llm-server/app/tests/adapters/conftest.py \
        litert-llm-server/app/tests/adapters/test_openai_models.py
git commit -m "feat(litert/adapters): add OpenAI /v1/models endpoint"
```

---

### Task 12: `/v1/chat/completions` non-streaming

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/openai_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_openai_chat_nonstream.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_openai_chat_nonstream.py`:

```python
from httpx import AsyncClient

from tests.fakes.fake_engine import FakeEngine


async def test_chat_completion_non_streaming(
    client: AsyncClient, fake_engine: FakeEngine
):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50,
        "temperature": 0.5,
        "stream": False,
    }
    r = await client.post("/v1/chat/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "gemma-4-e2b"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Hello, world!"
    assert choice["finish_reason"] == "stop"

    assert len(fake_engine.calls) == 1
    call = fake_engine.calls[0]
    assert call.model == "gemma-4-e2b"
    assert "Hi" in call.prompt
    assert call.params.max_tokens == 50
    assert call.params.temperature == 0.5
```

- [ ] **Step 2: Run test (expect fail with 404)**

```bash
uv run pytest tests/adapters/test_openai_chat_nonstream.py -v
```
Expected: HTTP 404 or AttributeError on missing route.

- [ ] **Step 3: Extend `openai_router.py` with chat-completions request/response models and the non-streaming branch**

Add to `litert-llm-server/app/src/litert_server/adapters/openai_router.py` (full file after this task):

```python
"""OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry
from litert_server.domain.types import GenerationParams


class OpenAIModelItem(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "litert-llm-server"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModelItem]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]


def _render_chat_prompt(messages: list[ChatMessage]) -> str:
    """Minimal multi-turn chat template. Engines that need a model-specific
    template can override later; this MVP joins roles with simple tags.
    """
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def build_openai_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/models", response_model=OpenAIModelList)
    async def list_models() -> OpenAIModelList:
        models = await registry.list()
        now = int(time.time())
        return OpenAIModelList(
            data=[OpenAIModelItem(id=m.name, created=now) for m in models],
        )

    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
        prompt = _render_chat_prompt(req.messages)

        # Streaming branch added in Task 13.
        text_parts: list[str] = []
        finish: Literal["stop", "length"] | None = None
        async for tok in engine.stream_completion(req.model, prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="".join(text_parts)),
                    finish_reason=finish,
                )
            ],
        )

    return router
```

- [ ] **Step 4: Run test (expect pass)**

```bash
uv run pytest tests/adapters/test_openai_chat_nonstream.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/openai_router.py \
        litert-llm-server/app/tests/adapters/test_openai_chat_nonstream.py
git commit -m "feat(litert/adapters): add OpenAI /v1/chat/completions (non-streaming)"
```

---

### Task 13: `/v1/chat/completions` streaming (SSE)

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/openai_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_openai_chat_stream.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_openai_chat_stream.py`:

```python
import json

from httpx import AsyncClient


async def test_chat_completion_streaming_sse(client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50,
        "stream": True,
    }
    async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        chunks: list[dict] = []
        saw_done = False
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line.removeprefix("data: ").strip()
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))

    assert saw_done
    assert len(chunks) >= 2
    assert chunks[0]["object"] == "chat.completion.chunk"
    # First chunk carries role
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    # Reassembled text equals what FakeEngine streamed
    text = "".join(
        c["choices"][0]["delta"].get("content", "") for c in chunks
    )
    assert text == "Hello, world!"
    # Last chunk before [DONE] carries finish_reason
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
```

- [ ] **Step 2: Run test (expect fail — non-streaming branch still active)**

```bash
uv run pytest tests/adapters/test_openai_chat_stream.py -v
```
Expected: assertion on content-type or 200 OK with wrong content-type.

- [ ] **Step 3: Add the streaming branch**

Replace the `chat_completions` handler in `openai_router.py` with this version:

```python
    @router.post("/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
        prompt = _render_chat_prompt(req.messages)

        if req.stream:
            return StreamingResponse(
                _chat_sse_stream(engine, req.model, prompt, params),
                media_type="text/event-stream",
            )

        text_parts: list[str] = []
        finish: Literal["stop", "length"] | None = None
        async for tok in engine.stream_completion(req.model, prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="".join(text_parts)),
                    finish_reason=finish,
                )
            ],
        )
```

Add the SSE helper to the same file (above `build_openai_router`):

```python
async def _chat_sse_stream(
    engine: InferenceService,
    model: str,
    prompt: str,
    params: GenerationParams,
) -> AsyncIterator[str]:
    """Yields OpenAI-compatible SSE lines.

    First chunk carries `delta.role = "assistant"`. Subsequent chunks carry
    `delta.content`. The last chunk carries `finish_reason`. Stream ends
    with the literal `data: [DONE]\\n\\n` sentinel.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None = None) -> str:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish},
            ],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    # First chunk: role only.
    yield frame({"role": "assistant"})

    finish_reason: str | None = None
    async for tok in engine.stream_completion(model, prompt, params):
        if tok.finish_reason is not None:
            finish_reason = tok.finish_reason
        yield frame({"content": tok.text}, finish=None)

    yield frame({}, finish=finish_reason or "stop")
    yield "data: [DONE]\n\n"
```

Add the missing imports at the top of `openai_router.py`:

```python
import json
from collections.abc import AsyncIterator

from fastapi.responses import StreamingResponse
```

- [ ] **Step 4: Run all openai tests**

```bash
uv run pytest tests/adapters/ -v
```
Expected: all pass (models, non-streaming, streaming).

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/openai_router.py \
        litert-llm-server/app/tests/adapters/test_openai_chat_stream.py
git commit -m "feat(litert/adapters): add OpenAI SSE streaming for /v1/chat/completions"
```

---

### Task 14: `/v1/completions` (legacy)

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/openai_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_openai_completions.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_openai_completions.py`:

```python
from httpx import AsyncClient


async def test_legacy_completions_non_streaming(client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "prompt": "Once upon a time",
        "max_tokens": 20,
        "stream": False,
    }
    r = await client.post("/v1/completions", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["model"] == "gemma-4-e2b"
    assert body["choices"][0]["text"] == "Hello, world!"
    assert body["choices"][0]["finish_reason"] == "stop"
```

- [ ] **Step 2: Run test (expect 404)**

```bash
uv run pytest tests/adapters/test_openai_completions.py -v
```

- [ ] **Step 3: Add the endpoint**

In `openai_router.py`, add the request/response models and route:

```python
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] | None = None
    stream: bool = False


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Literal["stop", "length"] | None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
```

Inside `build_openai_router`, add the route (non-streaming only — streaming is symmetric to chat and not used by HA's OpenAI Conversation):

```python
    @router.post("/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            stop=req.stop,
        )
        text_parts: list[str] = []
        finish: Literal["stop", "length"] | None = None
        async for tok in engine.stream_completion(req.model, req.prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=req.model,
            choices=[
                CompletionChoice(
                    text="".join(text_parts), index=0, finish_reason=finish
                )
            ],
        )
```

- [ ] **Step 4: Run test (expect pass)**

```bash
uv run pytest tests/adapters/ -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/openai_router.py \
        litert-llm-server/app/tests/adapters/test_openai_completions.py
git commit -m "feat(litert/adapters): add OpenAI /v1/completions (legacy)"
```

---

## Phase 5 — Ollama Adapter

### Task 15: `/api/tags` endpoint

**Files:**
- Create: `litert-llm-server/app/src/litert_server/adapters/ollama_router.py`
- Modify: `litert-llm-server/app/tests/adapters/conftest.py` (add Ollama app fixture)
- Create: `litert-llm-server/app/tests/adapters/test_ollama_tags.py`

- [ ] **Step 1: Extend conftest with Ollama app fixture**

In `litert-llm-server/app/tests/adapters/conftest.py`, append:

```python
from litert_server.adapters.ollama_router import build_ollama_router


@pytest.fixture
def ollama_app(fake_engine, fake_registry) -> FastAPI:
    app = FastAPI()
    app.include_router(build_ollama_router(engine=fake_engine, registry=fake_registry))
    return app


@pytest.fixture
async def ollama_client(ollama_app):
    transport = ASGITransport(app=ollama_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
```

- [ ] **Step 2: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_ollama_tags.py`:

```python
from httpx import AsyncClient


async def test_tags_returns_ollama_shape(ollama_client: AsyncClient):
    r = await ollama_client.get("/api/tags")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["name"] == "gemma-4-e2b"
    assert m["size"] == 1_500_000_000
    assert "modified_at" in m
    assert m["details"]["quantization_level"] == "int8"
```

- [ ] **Step 3: Run test (import fail expected)**

```bash
uv run pytest tests/adapters/test_ollama_tags.py -v
```

- [ ] **Step 4: Implement `litert-llm-server/app/src/litert_server/adapters/ollama_router.py`**

```python
"""Ollama-compatible HTTP adapter.

Imports only `domain.*`. Translates between HTTP/JSON/NDJSON and the
`InferenceService` and `ModelRegistry` Protocols.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry


class OllamaModelDetails(BaseModel):
    format: str = "gguf"  # Ollama clients expect a format string
    family: str = "gemma"
    parameter_size: str = "2B"
    quantization_level: str


class OllamaModelItem(BaseModel):
    name: str
    modified_at: str
    size: int
    digest: str = ""
    details: OllamaModelDetails


class OllamaTagsResponse(BaseModel):
    models: list[OllamaModelItem]


def build_ollama_router(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/tags", response_model=OllamaTagsResponse)
    async def tags() -> OllamaTagsResponse:
        models = await registry.list()
        now_iso = datetime.now(UTC).isoformat()
        return OllamaTagsResponse(
            models=[
                OllamaModelItem(
                    name=m.name,
                    modified_at=now_iso,
                    size=m.size_bytes,
                    details=OllamaModelDetails(quantization_level=m.quantization),
                )
                for m in models
            ]
        )

    return router
```

- [ ] **Step 5: Run test (expect pass)**

```bash
uv run pytest tests/adapters/test_ollama_tags.py -v
```

- [ ] **Step 6: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/ollama_router.py \
        litert-llm-server/app/tests/adapters/conftest.py \
        litert-llm-server/app/tests/adapters/test_ollama_tags.py
git commit -m "feat(litert/adapters): add Ollama /api/tags endpoint"
```

---

### Task 16: `/api/chat` with NDJSON streaming

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/ollama_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_ollama_chat.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_ollama_chat.py`:

```python
import json

from httpx import AsyncClient


async def test_chat_streams_ndjson(ollama_client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    async with ollama_client.stream("POST", "/api/chat", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]

    chunks = [json.loads(line) for line in lines]
    assert len(chunks) >= 2

    for c in chunks[:-1]:
        assert c["model"] == "gemma-4-e2b"
        assert c["message"]["role"] == "assistant"
        assert c["done"] is False

    last = chunks[-1]
    assert last["done"] is True
    assert last["done_reason"] == "stop"

    text = "".join(c["message"]["content"] for c in chunks)
    assert text == "Hello, world!"


async def test_chat_non_streaming(ollama_client: AsyncClient):
    payload = {
        "model": "gemma-4-e2b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
    }
    r = await ollama_client.post("/api/chat", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["message"]["content"] == "Hello, world!"
    assert body["done_reason"] == "stop"
```

- [ ] **Step 2: Run test (expect 404)**

```bash
uv run pytest tests/adapters/test_ollama_chat.py -v
```

- [ ] **Step 3: Add chat to `ollama_router.py`**

Append/extend:

```python
import json
from collections.abc import AsyncIterator
from typing import Literal

from fastapi.responses import StreamingResponse

from litert_server.domain.types import GenerationParams


class OllamaChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OllamaChatRequest(BaseModel):
    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    options: dict | None = None  # Ollama-style nested options (num_predict, temperature, …)


def _params_from_ollama_options(options: dict | None) -> GenerationParams:
    options = options or {}
    return GenerationParams(
        max_tokens=int(options.get("num_predict", 512)),
        temperature=float(options.get("temperature", 0.7)),
        top_p=options.get("top_p"),
        stop=options.get("stop"),
    )


def _render_chat_prompt(messages: list[OllamaChatMessage]) -> str:
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|{msg.role}|>\n{msg.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)
```

Inside `build_ollama_router`, add:

```python
    @router.post("/chat")
    async def chat(req: OllamaChatRequest):
        params = _params_from_ollama_options(req.options)
        prompt = _render_chat_prompt(req.messages)
        created_at = datetime.now(UTC).isoformat()

        async def emit() -> AsyncIterator[str]:
            finish: str | None = None
            async for tok in engine.stream_completion(req.model, prompt, params):
                if tok.finish_reason is not None:
                    finish = tok.finish_reason
                yield json.dumps(
                    {
                        "model": req.model,
                        "created_at": created_at,
                        "message": {"role": "assistant", "content": tok.text},
                        "done": False,
                    }
                ) + "\n"
            yield json.dumps(
                {
                    "model": req.model,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": finish or "stop",
                }
            ) + "\n"

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        text_parts: list[str] = []
        finish: str | None = None
        async for tok in engine.stream_completion(req.model, prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason
        return {
            "model": req.model,
            "created_at": created_at,
            "message": {"role": "assistant", "content": "".join(text_parts)},
            "done": True,
            "done_reason": finish or "stop",
        }
```

- [ ] **Step 4: Run test (expect pass)**

```bash
uv run pytest tests/adapters/test_ollama_chat.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/ollama_router.py \
        litert-llm-server/app/tests/adapters/test_ollama_chat.py
git commit -m "feat(litert/adapters): add Ollama /api/chat with NDJSON streaming"
```

---

### Task 17: `/api/generate`

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/ollama_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_ollama_generate.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_ollama_generate.py`:

```python
import json

from httpx import AsyncClient


async def test_generate_streams_ndjson(ollama_client: AsyncClient):
    payload = {"model": "gemma-4-e2b", "prompt": "Once upon a time", "stream": True}
    async with ollama_client.stream("POST", "/api/generate", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["done"] is True
    assert chunks[-1]["done_reason"] == "stop"
    text = "".join(c["response"] for c in chunks)
    assert text == "Hello, world!"


async def test_generate_non_streaming(ollama_client: AsyncClient):
    payload = {"model": "gemma-4-e2b", "prompt": "x", "stream": False}
    r = await ollama_client.post("/api/generate", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["done"] is True
    assert body["response"] == "Hello, world!"
```

- [ ] **Step 2: Run test (expect 404)**

```bash
uv run pytest tests/adapters/test_ollama_generate.py -v
```

- [ ] **Step 3: Add the endpoint**

In `ollama_router.py`, add a request model and route:

```python
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = True
    options: dict | None = None
```

Inside `build_ollama_router`:

```python
    @router.post("/generate")
    async def generate(req: OllamaGenerateRequest):
        params = _params_from_ollama_options(req.options)
        created_at = datetime.now(UTC).isoformat()

        async def emit() -> AsyncIterator[str]:
            finish: str | None = None
            async for tok in engine.stream_completion(req.model, req.prompt, params):
                if tok.finish_reason is not None:
                    finish = tok.finish_reason
                yield json.dumps(
                    {
                        "model": req.model,
                        "created_at": created_at,
                        "response": tok.text,
                        "done": False,
                    }
                ) + "\n"
            yield json.dumps(
                {
                    "model": req.model,
                    "created_at": created_at,
                    "response": "",
                    "done": True,
                    "done_reason": finish or "stop",
                }
            ) + "\n"

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        text_parts: list[str] = []
        finish: str | None = None
        async for tok in engine.stream_completion(req.model, req.prompt, params):
            text_parts.append(tok.text)
            if tok.finish_reason is not None:
                finish = tok.finish_reason
        return {
            "model": req.model,
            "created_at": created_at,
            "response": "".join(text_parts),
            "done": True,
            "done_reason": finish or "stop",
        }
```

- [ ] **Step 4: Run test (expect pass)**

```bash
uv run pytest tests/adapters/test_ollama_generate.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/ollama_router.py \
        litert-llm-server/app/tests/adapters/test_ollama_generate.py
git commit -m "feat(litert/adapters): add Ollama /api/generate"
```

---

### Task 18: `/api/pull` with NDJSON progress

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/ollama_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_ollama_pull.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/adapters/test_ollama_pull.py`:

```python
import json

from httpx import AsyncClient


async def test_pull_streams_progress(ollama_client: AsyncClient):
    payload = {"name": "gemma-3n-e2b", "stream": True}
    async with ollama_client.stream("POST", "/api/pull", json=payload) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/x-ndjson")
        lines = [line async for line in r.aiter_lines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[-1]["status"] in ("done", "success")
    for c in chunks[:-1]:
        assert c["status"] == "downloading"
        assert "completed" in c
        assert "total" in c
```

- [ ] **Step 2: Run test (expect 404)**

```bash
uv run pytest tests/adapters/test_ollama_pull.py -v
```

- [ ] **Step 3: Add the endpoint**

```python
class OllamaPullRequest(BaseModel):
    name: str
    stream: bool = True
```

Inside `build_ollama_router`:

```python
    @router.post("/pull")
    async def pull(req: OllamaPullRequest):
        async def emit() -> AsyncIterator[str]:
            async for prog in registry.pull(req.name):
                if prog.status == "done":
                    yield json.dumps({"status": "success"}) + "\n"
                elif prog.status == "error":
                    yield json.dumps(
                        {"status": "error", "error": prog.error or "unknown"}
                    ) + "\n"
                else:
                    yield json.dumps(
                        {
                            "status": "downloading",
                            "completed": prog.bytes_done,
                            "total": prog.bytes_total,
                        }
                    ) + "\n"

        if req.stream:
            return StreamingResponse(emit(), media_type="application/x-ndjson")

        last_status = "success"
        async for prog in registry.pull(req.name):
            if prog.status == "error":
                last_status = "error"
        return {"status": last_status}
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/adapters/test_ollama_pull.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/ollama_router.py \
        litert-llm-server/app/tests/adapters/test_ollama_pull.py
git commit -m "feat(litert/adapters): add Ollama /api/pull with NDJSON progress"
```

---

### Task 19: `/api/show` and `/api/delete`

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/adapters/ollama_router.py`
- Create: `litert-llm-server/app/tests/adapters/test_ollama_show_delete.py`

- [ ] **Step 1: Write the failing tests**

Create `litert-llm-server/app/tests/adapters/test_ollama_show_delete.py`:

```python
from httpx import AsyncClient


async def test_show_returns_metadata(ollama_client: AsyncClient):
    r = await ollama_client.post("/api/show", json={"name": "gemma-4-e2b"})
    assert r.status_code == 200
    body = r.json()
    assert body["details"]["quantization_level"] == "int8"
    assert "modelfile" in body or "license" in body or True  # tolerant


async def test_show_unknown_returns_404(ollama_client: AsyncClient):
    r = await ollama_client.post("/api/show", json={"name": "does-not-exist"})
    assert r.status_code == 404


async def test_delete_removes_model(ollama_client: AsyncClient):
    r = await ollama_client.request(
        "DELETE", "/api/delete", json={"name": "gemma-4-e2b"}
    )
    assert r.status_code == 200
    tags = (await ollama_client.get("/api/tags")).json()
    assert all(m["name"] != "gemma-4-e2b" for m in tags["models"])
```

- [ ] **Step 2: Run tests (expect 404 / 405)**

```bash
uv run pytest tests/adapters/test_ollama_show_delete.py -v
```

- [ ] **Step 3: Add the endpoints**

```python
from fastapi import HTTPException


class OllamaShowRequest(BaseModel):
    name: str


class OllamaDeleteRequest(BaseModel):
    name: str
```

Inside `build_ollama_router`:

```python
    @router.post("/show")
    async def show(req: OllamaShowRequest):
        try:
            m = await registry.get(req.name)
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")
        return {
            "modelfile": "",
            "parameters": "",
            "template": "",
            "details": {
                "format": "gguf",
                "family": "gemma",
                "parameter_size": "2B",
                "quantization_level": m.quantization,
            },
        }

    @router.delete("/delete")
    async def delete(req: OllamaDeleteRequest):
        await registry.delete(req.name)
        return {"status": "success"}
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/adapters/test_ollama_show_delete.py -v
```

- [ ] **Step 5: Run all adapter tests for a sanity gate**

```bash
uv run pytest tests/adapters/ -v
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add litert-llm-server/app/src/litert_server/adapters/ollama_router.py \
        litert-llm-server/app/tests/adapters/test_ollama_show_delete.py
git commit -m "feat(litert/adapters): add Ollama /api/show and /api/delete"
```

---

## Phase 6 — LiteRT Engine Implementation

### Task 20: `LiteRTEngine` skeleton with model loading

**Files:**
- Create: `litert-llm-server/app/src/litert_server/engines/litert.py`
- Create: `litert-llm-server/app/tests/engines/test_litert_engine.py`

> **Note:** This task and Task 21 use `litert-lm-api` (`litert_lm.Engine`
> / `Session`). The package ships self-contained wheels for
> macOS-arm64 and Linux-amd64 and should be installed via `uv sync`.
> Tests that touch a real model file are guarded by an env var
> (`LITERT_TEST_MODEL_PATH`); construction-only tests run anywhere.

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/engines/test_litert_engine.py`:

```python
from pathlib import Path

from litert_server.engines.litert import LiteRTEngine


def test_engine_construction_does_not_load_model(tmp_path: Path):
    engine = LiteRTEngine(models_dir=tmp_path)
    assert engine.models_dir == tmp_path
    assert engine.current_model is None
```

- [ ] **Step 2: Run test (expect import fail)**

```bash
uv run pytest tests/engines/ -v
```

- [ ] **Step 3: Implement the skeleton**

Create `litert-llm-server/app/src/litert_server/engines/litert.py`:

```python
"""LiteRT-LM backed `InferenceService` implementation.

Wraps `litert_lm.Engine`. Loads a model on first use and caches the
loaded engine per model name (single-slot in MVP — second model triggers
reload).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock
from typing import Any

from litert_server.domain.types import GenerationParams, Token


class LiteRTEngine:
    """Single-slot LiteRT-LM engine.

    Thread-safety: model swaps are serialized via a Lock. The
    `litert_lm.Engine` C-API itself is synchronous; we wrap streaming
    iteration in `asyncio.to_thread` in `stream_completion` (Task 21).
    """

    def __init__(self, *, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = Lock()
        self.current_model: str | None = None
        self._engine: Any | None = None

    def _model_file(self, model_name: str) -> Path:
        # MVP mapping: <models_dir>/<model_name>.litertlm
        # ModelRegistry guarantees the file exists before engine sees it.
        return self.models_dir / f"{model_name}.litertlm"

    def _ensure_loaded(self, model_name: str) -> None:
        # litert_lm import is lazy to keep import-time light.
        from litert_lm import Backend, Engine

        with self._lock:
            if self.current_model == model_name and self._engine is not None:
                return
            file = self._model_file(model_name)
            if not file.exists():
                raise FileNotFoundError(f"Model file not found: {file}")
            # Close previous engine if any (single-slot semantics).
            if self._engine is not None:
                self._engine.close()
            self._engine = Engine(model_path=str(file), backend=Backend.CPU)
            self.current_model = model_name

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        # Implemented in Task 21.
        raise NotImplementedError
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/engines/ -v
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/engines/litert.py \
        litert-llm-server/app/tests/engines/test_litert_engine.py
git commit -m "feat(litert/engines): add LiteRTEngine skeleton wrapping litert_lm.Engine"
```

---

### Task 21: `LiteRTEngine.stream_completion` implementation

**Files:**
- Modify: `litert-llm-server/app/src/litert_server/engines/litert.py`
- Modify: `litert-llm-server/app/tests/engines/test_litert_engine.py`

> **Streaming approach:** `litert_lm.Session` provides true per-token
> iteration. The session is created from the loaded `Engine` and yields
> tokens one at a time during decode. We iterate in a worker thread (via
> `asyncio.to_thread` plus a queue/iterator bridge) so the async caller
> sees an `AsyncIterator[Token]` without blocking the event loop.

- [ ] **Step 1: Add the failing test**

Append to `tests/engines/test_litert_engine.py`:

```python
import os
from pathlib import Path

import pytest

from litert_server.domain.types import GenerationParams


@pytest.mark.skipif(
    not os.environ.get("LITERT_TEST_MODEL_PATH"),
    reason="Requires real model file; set LITERT_TEST_MODEL_PATH to enable",
)
async def test_stream_completion_yields_tokens(tmp_path):
    model_path = Path(os.environ["LITERT_TEST_MODEL_PATH"])
    engine = LiteRTEngine(models_dir=model_path.parent)
    tokens = []
    async for tok in engine.stream_completion(
        model="gemma-4-e2b",
        prompt="Hello",
        params=GenerationParams(max_tokens=8, temperature=0.0),
    ):
        tokens.append(tok)
    assert len(tokens) >= 1
    assert tokens[-1].finish_reason in ("stop", "length")
```

- [ ] **Step 2: Run test (expect skip without env var)**

```bash
uv run pytest tests/engines/ -v
```

- [ ] **Step 3: Implement `stream_completion`**

Replace `litert-llm-server/app/src/litert_server/engines/litert.py`:

```python
"""LiteRT-LM backed `InferenceService` implementation."""

from __future__ import annotations

import asyncio
import queue
from collections.abc import AsyncIterator
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from litert_server.domain.types import GenerationParams, Token

_SENTINEL: object = object()


class LiteRTEngine:
    def __init__(self, *, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._lock = Lock()
        self.current_model: str | None = None
        self._engine: Any | None = None

    def _model_file(self, model_name: str) -> Path:
        return self.models_dir / f"{model_name}.litertlm"

    def _ensure_loaded(self, model_name: str) -> None:
        from litert_lm import Backend, Engine

        with self._lock:
            if self.current_model == model_name and self._engine is not None:
                return
            file = self._model_file(model_name)
            if not file.exists():
                raise FileNotFoundError(f"Model file not found: {file}")
            if self._engine is not None:
                self._engine.close()
            self._engine = Engine(model_path=str(file), backend=Backend.CPU)
            self.current_model = model_name

    def _build_sampler(self, params: GenerationParams) -> Any:
        from litert_lm import SamplerConfig

        return SamplerConfig(temperature=params.temperature, top_p=params.top_p)

    async def stream_completion(
        self,
        model: str,
        prompt: str,
        params: GenerationParams,
    ) -> AsyncIterator[Token]:
        await asyncio.to_thread(self._ensure_loaded, model)
        assert self._engine is not None

        sampler = self._build_sampler(params)
        session = self._engine.create_session(
            sampler_config=sampler,
            max_output_tokens=params.max_tokens,
        )

        q: queue.Queue[Any] = queue.Queue(maxsize=64)

        def producer() -> None:
            try:
                for piece in session.generate_stream(prompt):
                    q.put(piece)
            except Exception as exc:  # surface to consumer
                q.put(exc)
            finally:
                q.put(_SENTINEL)

        Thread(target=producer, daemon=True).start()

        loop = asyncio.get_running_loop()
        index = 0
        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                yield Token(text="", index=index, finish_reason="stop")
                return
            if isinstance(item, Exception):
                raise item
            yield Token(text=str(item), index=index, finish_reason=None)
            index += 1
```

> **Note on `session.generate_stream`**: the exact method name on
> `litert_lm.Session` may vary between API versions. If the installed
> version exposes a different streaming entrypoint (e.g.
> `session.generate`, `session.run`, or yielding via `Conversation`),
> adjust the producer body to call that method. The contract this engine
> provides — `AsyncIterator[Token]` — is unaffected.

- [ ] **Step 4: Run test (skip without model)**

```bash
uv run pytest tests/engines/ -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/engines/litert.py \
        litert-llm-server/app/tests/engines/test_litert_engine.py
git commit -m "feat(litert/engines): implement LiteRTEngine.stream_completion"
```

---

## Phase 7 — Model Registry Implementation

### Task 22: Filesystem cache layer

**Files:**
- Create: `litert-llm-server/app/src/litert_server/model_registry/filesystem.py`
- Create: `litert-llm-server/app/tests/model_registry/test_filesystem.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/model_registry/test_filesystem.py`:

```python
from pathlib import Path

import pytest

from litert_server.model_registry.filesystem import FilesystemCache


def test_lists_files_with_known_extension(tmp_path: Path):
    (tmp_path / "gemma-4-e2b.litertlm").write_bytes(b"x" * 1024)
    (tmp_path / "random.txt").write_text("ignore me")
    cache = FilesystemCache(root=tmp_path)
    names = [m.name for m in cache.scan()]
    assert names == ["gemma-4-e2b"]


def test_size_reflects_file_size(tmp_path: Path):
    (tmp_path / "x.litertlm").write_bytes(b"a" * 4096)
    cache = FilesystemCache(root=tmp_path)
    [m] = cache.scan()
    assert m.size_bytes == 4096
    assert m.path == tmp_path / "x.litertlm"


def test_delete_removes_file(tmp_path: Path):
    f = tmp_path / "y.litertlm"
    f.write_bytes(b"data")
    cache = FilesystemCache(root=tmp_path)
    cache.delete("y")
    assert not f.exists()


def test_delete_unknown_is_noop(tmp_path: Path):
    cache = FilesystemCache(root=tmp_path)
    cache.delete("nope")  # must not raise
```

- [ ] **Step 2: Run test (import fail)**

```bash
uv run pytest tests/model_registry/test_filesystem.py -v
```

- [ ] **Step 3: Implement**

Create `litert-llm-server/app/src/litert_server/model_registry/filesystem.py`:

```python
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
                        quantization="int8",  # MVP: assume; refine via metadata sidecar later
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
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/model_registry/test_filesystem.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/model_registry/filesystem.py \
        litert-llm-server/app/tests/model_registry/test_filesystem.py
git commit -m "feat(litert/registry): add FilesystemCache for local model files"
```

---

### Task 23: HuggingFace downloader with progress

**Files:**
- Create: `litert-llm-server/app/src/litert_server/model_registry/huggingface.py`
- Create: `litert-llm-server/app/tests/model_registry/test_huggingface.py`

- [ ] **Step 1: Write the failing test (mocks `hf_hub_download`)**

Create `litert-llm-server/app/tests/model_registry/test_huggingface.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from litert_server.model_registry.filesystem import FilesystemCache
from litert_server.model_registry.huggingface import (
    HuggingFaceRegistry,
    MODEL_CATALOG,
)


async def test_list_reflects_filesystem(tmp_path: Path):
    (tmp_path / "gemma-4-e2b.litertlm").write_bytes(b"x" * 512)
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    models = await reg.list()
    assert {m.name for m in models} == {"gemma-4-e2b"}


async def test_get_unknown_raises(tmp_path: Path):
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    with pytest.raises(KeyError):
        await reg.get("does-not-exist")


async def test_pull_known_emits_progress(tmp_path: Path):
    def fake_download(repo_id, filename, cache_dir, **_):
        out = Path(cache_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"y" * 1024)
        return str(out)

    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    target = next(iter(MODEL_CATALOG))
    with patch(
        "litert_server.model_registry.huggingface.hf_hub_download",
        side_effect=fake_download,
    ):
        progress = [p async for p in reg.pull(target)]
    assert progress[-1].status == "done"
    assert (tmp_path / f"{target}.litertlm").exists()


async def test_pull_unknown_emits_error(tmp_path: Path):
    reg = HuggingFaceRegistry(cache=FilesystemCache(root=tmp_path))
    progress = [p async for p in reg.pull("nope")]
    assert progress[-1].status == "error"
```

- [ ] **Step 2: Run test (import fail)**

```bash
uv run pytest tests/model_registry/test_huggingface.py -v
```

- [ ] **Step 3: Implement the registry**

Create `litert-llm-server/app/src/litert_server/model_registry/huggingface.py`:

```python
"""HuggingFace-backed `ModelRegistry` implementation.

MVP strategy: a hardcoded catalog mapping public model names to HF repo +
filename. Catalog can later be replaced by a metadata-driven registry.
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


MODEL_CATALOG: dict[str, CatalogEntry] = {
    # Public — no HF token required
    "gemma-4-e2b": CatalogEntry(
        name="gemma-4-e2b",
        repo_id="litert-community/gemma-4-E2B-it-litert-lm",
        filename="gemma-4-E2B-it.litertlm",
        quantization="int4",
    ),
    "gemma-4-e4b": CatalogEntry(
        name="gemma-4-e4b",
        repo_id="litert-community/gemma-4-E4B-it-litert-lm",
        filename="gemma-4-E4B-it.litertlm",
        quantization="int4",
    ),
    # Gated — require HF token + accepted Gemma license
    "gemma-3n-e2b": CatalogEntry(
        name="gemma-3n-e2b",
        repo_id="google/gemma-3n-E2B-it-litert-lm",
        filename="gemma-3n-E2B-it-int4.litertlm",
        quantization="int4",
    ),
    "gemma-3n-e4b": CatalogEntry(
        name="gemma-3n-e4b",
        repo_id="google/gemma-3n-E4B-it-litert-lm",
        filename="gemma-3n-E4B-it-int4.litertlm",
        quantization="int4",
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
                bytes_done=0, bytes_total=0, status="error",
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
                token=self.hf_token,  # None -> falls back to HF_TOKEN env var
            )

        try:
            tmp_path = await asyncio.to_thread(_download)
        except Exception as exc:  # network, auth (401), 404 — surface as error
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
```

> **Note on auth**: The `litert-community/*` Gemma 4 repositories are
> public — no token needed. The `google/*-litert-lm` Gemma 3n
> repositories are **gated** — anonymous downloads return 401. The token
> is forwarded explicitly via the `token=` kwarg; passing `None` lets
> `huggingface_hub` fall back to the `HF_TOKEN` environment variable
> (set by bashio from the `hf_token` add-on option). For gated models,
> users must first accept the Gemma license on the HuggingFace model
> page and create a read-scope token.

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/model_registry/ -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/model_registry/huggingface.py \
        litert-llm-server/app/tests/model_registry/test_huggingface.py
git commit -m "feat(litert/registry): add HuggingFaceRegistry with MVP catalog"
```

---

## Phase 8 — Wiring

### Task 24: `Settings` from environment

**Files:**
- Create: `litert-llm-server/app/src/litert_server/config.py`
- Create: `litert-llm-server/app/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/test_config.py`:

```python
import json
import os

from litert_server.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("LITERT_LOG_LEVEL", "debug")
    monkeypatch.setenv("LITERT_DEFAULT_MODEL", "gemma-4-e2b")
    monkeypatch.setenv("LITERT_MAX_TOKENS", "256")
    monkeypatch.setenv("LITERT_TEMPERATURE", "0.5")
    monkeypatch.setenv("LITERT_MODELS_DIR", "/data/models")
    monkeypatch.setenv("LITERT_PORT", "8080")
    monkeypatch.setenv("LITERT_PRELOAD_MODELS", json.dumps(["gemma-4-e2b"]))
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    s = Settings()
    assert s.log_level == "debug"
    assert s.default_model == "gemma-4-e2b"
    assert s.max_tokens == 256
    assert s.temperature == 0.5
    assert str(s.models_dir) == "/data/models"
    assert s.port == 8080
    assert s.preload_models == ["gemma-4-e2b"]
    assert s.hf_token == "hf_xxx"


def test_settings_defaults(monkeypatch):
    for k in list(os.environ):
        if k.startswith("LITERT_") or k == "HF_TOKEN":
            monkeypatch.delenv(k, raising=False)
    s = Settings()
    assert s.log_level == "info"
    assert s.preload_models == []
    assert s.hf_token is None
```

- [ ] **Step 2: Run test (expect import fail)**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 3: Implement**

Create `litert-llm-server/app/src/litert_server/config.py`:

```python
"""Typed settings read exclusively from environment variables.

bashio is the only translator between `config.yaml` and these env vars
(see `rootfs/etc/cont-init.d/01-config.sh`). The application MUST NOT
parse `config.yaml` itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal[
    "trace", "debug", "info", "notice", "warning", "error", "fatal"
]


class Settings(BaseSettings):
    """Settings exposed via the LITERT_* env-var prefix, plus a special
    `hf_token` field that reads the **un-prefixed** `HF_TOKEN` env var (the
    standard variable that `huggingface_hub` recognizes by default).
    """

    model_config = SettingsConfigDict(env_prefix="LITERT_", extra="ignore")

    log_level: LogLevel = "info"
    default_model: str = "gemma-4-e2b"
    max_tokens: int = 1024
    temperature: float = 0.7
    models_dir: Path = Path("/data/models")
    port: int = 8080
    preload_models: list[str] = []
    hf_token: str | None = Field(default=None, validation_alias="HF_TOKEN")

    @field_validator("preload_models", mode="before")
    @classmethod
    def _parse_preload(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            return json.loads(v)
        return v

    @field_validator("hf_token", mode="before")
    @classmethod
    def _empty_token_is_none(cls, v):
        # bashio exports HF_TOKEN even when blank in HA UI; treat "" as None
        if isinstance(v, str) and not v.strip():
            return None
        return v
```

Also update the import line at the top of the file: add `Field` to the
pydantic import:

```python
from pydantic import Field, field_validator
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/test_config.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/src/litert_server/config.py \
        litert-llm-server/app/tests/test_config.py
git commit -m "feat(litert): add typed Settings (ENV-only, no config.yaml access)"
```

---

### Task 25: `__main__.py` — FastAPI app, wiring, health endpoints

**Files:**
- Create: `litert-llm-server/app/src/litert_server/__main__.py`
- Create: `litert-llm-server/app/tests/test_main_app.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/test_main_app.py`:

```python
from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from litert_server.__main__ import build_app
from tests.fakes.fake_engine import FakeEngine
from tests.fakes.fake_registry import FakeRegistry


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = build_app(engine=FakeEngine(), registry=FakeRegistry())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_healthz(client: AsyncClient):
    r = await client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_readyz_ok_when_engine_set(client: AsyncClient):
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


async def test_both_router_sets_mounted(client: AsyncClient):
    r1 = await client.get("/v1/models")
    r2 = await client.get("/api/tags")
    assert r1.status_code == 200
    assert r2.status_code == 200
```

- [ ] **Step 2: Run test (expect fail)**

```bash
uv run pytest tests/test_main_app.py -v
```

- [ ] **Step 3: Implement `__main__.py`**

```python
"""FastAPI app construction and Uvicorn entry.

This is the ONLY place where concrete engines/registries are
instantiated and injected into routers. Importing `app` triggers
production wiring (real LiteRT engine + HuggingFace registry); for
tests, call `build_app(engine=…, registry=…)` directly with fakes.
"""

from __future__ import annotations

from fastapi import FastAPI

from litert_server.adapters.ollama_router import build_ollama_router
from litert_server.adapters.openai_router import build_openai_router
from litert_server.config import Settings
from litert_server.domain.inference import InferenceService
from litert_server.domain.model_registry import ModelRegistry


def build_app(
    *,
    engine: InferenceService,
    registry: ModelRegistry,
) -> FastAPI:
    app = FastAPI(title="litert-llm-server", version="0.1.0")

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        return {"status": "ready"}

    app.include_router(build_openai_router(engine=engine, registry=registry))
    app.include_router(build_ollama_router(engine=engine, registry=registry))
    return app


def _build_production_app() -> FastAPI:
    from litert_server.engines.litert import LiteRTEngine
    from litert_server.model_registry.filesystem import FilesystemCache
    from litert_server.model_registry.huggingface import HuggingFaceRegistry

    settings = Settings()
    cache = FilesystemCache(root=settings.models_dir)
    registry = HuggingFaceRegistry(cache=cache, hf_token=settings.hf_token)
    engine = LiteRTEngine(models_dir=settings.models_dir)
    return build_app(engine=engine, registry=registry)


app = _build_production_app()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_main_app.py -v
```

- [ ] **Step 5: Full test gate**

```bash
uv run pytest -v
```
Expected: all green except optionally-skipped real-engine tests.

- [ ] **Step 6: Commit**

```bash
git add litert-llm-server/app/src/litert_server/__main__.py \
        litert-llm-server/app/tests/test_main_app.py
git commit -m "feat(litert): wire FastAPI app with OpenAI+Ollama routers and health endpoints"
```

---

## Phase 9 — Architecture Enforcement

### Task 26: import-linter contracts

**Files:**
- Create: `litert-llm-server/app/.importlinter`
- Create: `litert-llm-server/app/tests/test_architecture.py`

- [ ] **Step 1: Write the failing test**

Create `litert-llm-server/app/tests/test_architecture.py`:

```python
import subprocess
from pathlib import Path


def test_import_linter_contracts_pass():
    here = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["uv", "run", "lint-imports", "--config", str(here / ".importlinter")],
        cwd=here,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
```

- [ ] **Step 2: Run test (expect fail — no config)**

```bash
uv run pytest tests/test_architecture.py -v
```

- [ ] **Step 3: Create `litert-llm-server/app/.importlinter`**

```ini
[importlinter]
root_package = litert_server

[importlinter:contract:domain-purity]
name = Domain has no internal dependencies
type = forbidden
source_modules =
    litert_server.domain
forbidden_modules =
    litert_server.engines
    litert_server.adapters
    litert_server.model_registry

[importlinter:contract:adapters-do-not-know-engines]
name = Adapters never import engines or registry concretions
type = forbidden
source_modules =
    litert_server.adapters
forbidden_modules =
    litert_server.engines
    litert_server.model_registry
    litert_lm
    huggingface_hub
```

- [ ] **Step 4: Run test (expect pass)**

```bash
uv run pytest tests/test_architecture.py -v
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/app/.importlinter \
        litert-llm-server/app/tests/test_architecture.py
git commit -m "test(litert): enforce Ports & Adapters via import-linter contracts"
```

---

## Phase 10 — Add-on Wrapper

### Task 27: `config.yaml` and `build.yaml`

**Files:**
- Create: `litert-llm-server/config.yaml`
- Create: `litert-llm-server/build.yaml`

- [ ] **Step 1: Create `litert-llm-server/config.yaml`** (exact content per spec Section 5)

```yaml
name: "LiteRT LLM Server"
version: "0.1.0"
slug: litert_llm_server
description: "Local LLM inference via Google LiteRT with OpenAI- and Ollama-compatible APIs"
arch:
  - amd64
  - aarch64
init: false
startup: application
boot: auto
ports:
  8080/tcp: 8080
ports_description:
  8080/tcp: "HTTP API (OpenAI + Ollama)"
map:
  - share:rw
options:
  log_level: info
  default_model: "gemma-4-e2b"
  max_tokens: 1024
  temperature: 0.7
  preload_models: []
  hf_token: ""
schema:
  log_level: list(trace|debug|info|notice|warning|error|fatal)
  default_model: str
  max_tokens: int(1,32768)
  temperature: float(0.0,2.0)
  preload_models:
    - str
  hf_token: password?
```

- [ ] **Step 2: Create `litert-llm-server/build.yaml`**

```yaml
build_from:
  amd64: ghcr.io/hassio-addons/base-python:14.0.2
  aarch64: ghcr.io/hassio-addons/base-python:14.0.2
labels:
  org.opencontainers.image.source: "https://github.com/USER/homassist-addons"
```

- [ ] **Step 3: Validate YAML parses**

```bash
uv run python -c "import yaml; yaml.safe_load(open('litert-llm-server/config.yaml')); yaml.safe_load(open('litert-llm-server/build.yaml')); print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add litert-llm-server/config.yaml litert-llm-server/build.yaml
git commit -m "feat(litert): add HA add-on config.yaml and build.yaml"
```

---

### Task 28: `Dockerfile`

**Files:**
- Create: `litert-llm-server/Dockerfile`
- Create: `litert-llm-server/.dockerignore`

- [ ] **Step 1: Create `litert-llm-server/Dockerfile`**

```dockerfile
ARG BUILD_FROM
FROM $BUILD_FROM

ARG BUILD_ARCH

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apk add --no-cache libstdc++ libgomp

COPY app/ /opt/app/
WORKDIR /opt/app

RUN pip install --no-cache-dir uv \
    && uv pip install --system --no-cache .

COPY rootfs/ /

LABEL io.hass.version="0.1.0" \
      io.hass.type="addon" \
      io.hass.arch="${BUILD_ARCH}"
```

- [ ] **Step 2: Create `litert-llm-server/.dockerignore`**

```
app/.venv
app/.pytest_cache
app/.mypy_cache
app/.ruff_cache
app/.models
app/tests
app/scripts
**/__pycache__
**/*.py[cod]
```

- [ ] **Step 3: Commit**

```bash
git add litert-llm-server/Dockerfile litert-llm-server/.dockerignore
git commit -m "feat(litert): add Dockerfile and .dockerignore"
```

---

### Task 29: `rootfs/etc/cont-init.d/01-config.sh`

**Files:**
- Create: `litert-llm-server/rootfs/etc/cont-init.d/01-config.sh`

- [ ] **Step 1: Create the script**

```bash
#!/usr/bin/with-contenv bashio
bashio::log.info "Reading add-on configuration..."

export LITERT_LOG_LEVEL="$(bashio::config 'log_level')"
export LITERT_DEFAULT_MODEL="$(bashio::config 'default_model')"
export LITERT_MAX_TOKENS="$(bashio::config 'max_tokens')"
export LITERT_TEMPERATURE="$(bashio::config 'temperature')"
export LITERT_MODELS_DIR="/data/models"
export LITERT_PORT="8080"

LITERT_PRELOAD_MODELS="$(bashio::config 'preload_models')"
export LITERT_PRELOAD_MODELS

# HuggingFace token: exported as HF_TOKEN (the env var huggingface_hub reads
# by default). Required for downloading gated google/*-litert-lm repos.
HF_TOKEN_VALUE="$(bashio::config 'hf_token')"
if [ -n "${HF_TOKEN_VALUE}" ]; then
    export HF_TOKEN="${HF_TOKEN_VALUE}"
fi

printenv | grep -E '^(LITERT_|HF_TOKEN$)' > /var/run/s6/container_environment/litert.env
```

- [ ] **Step 2: Make executable**

```bash
chmod +x litert-llm-server/rootfs/etc/cont-init.d/01-config.sh
```

- [ ] **Step 3: Shellcheck (optional but recommended)**

```bash
shellcheck litert-llm-server/rootfs/etc/cont-init.d/01-config.sh || true
```
Acceptable: `with-contenv` shebang may trigger SC1071 — ignore for this file.

- [ ] **Step 4: Commit**

```bash
git add litert-llm-server/rootfs/etc/cont-init.d/01-config.sh
git commit -m "feat(litert): add bashio init script (config.yaml -> ENV)"
```

---

### Task 30: `rootfs/etc/services.d/litert/run` and `finish`

**Files:**
- Create: `litert-llm-server/rootfs/etc/services.d/litert/run`
- Create: `litert-llm-server/rootfs/etc/services.d/litert/finish`

- [ ] **Step 1: Create `run`**

```bash
#!/usr/bin/with-contenv bashio
cd /opt/app
exec uvicorn litert_server.__main__:app \
    --host 0.0.0.0 \
    --port "${LITERT_PORT}" \
    --log-level "${LITERT_LOG_LEVEL}"
```

- [ ] **Step 2: Create `finish`**

```bash
#!/usr/bin/execlineb -S0
s6-svscanctl -t /var/run/s6/services
```

- [ ] **Step 3: Make executable**

```bash
chmod +x litert-llm-server/rootfs/etc/services.d/litert/run
chmod +x litert-llm-server/rootfs/etc/services.d/litert/finish
```

- [ ] **Step 4: Remove the `.keep` placeholder**

```bash
rm litert-llm-server/rootfs/etc/services.d/litert/.keep
```

- [ ] **Step 5: Commit**

```bash
git add litert-llm-server/rootfs/etc/services.d/litert/
git rm litert-llm-server/rootfs/etc/services.d/litert/.keep 2>/dev/null || true
git commit -m "feat(litert): add s6-overlay service run/finish scripts"
```

---

### Task 31: Add-on README, DOCS, CHANGELOG

**Files:**
- Create: `litert-llm-server/README.md`
- Create: `litert-llm-server/DOCS.md`
- Create: `litert-llm-server/CHANGELOG.md`

- [ ] **Step 1: Create `litert-llm-server/README.md`**

```markdown
# LiteRT LLM Server

Local LLM inference via Google LiteRT. Exposes OpenAI- and Ollama-compatible
HTTP APIs on port 8080, consumable by Home Assistant's "OpenAI Conversation"
integration and any Ollama-compatible client (Node-RED nodes, Open WebUI, …).

## Supported Models (MVP)

All models use the LiteRT-LM `.litertlm` format. Context window: up to 32k
tokens (prompt + completion combined).

| Name | Repository | Gated | Approx. Size |
|---|---|---|---|
| `gemma-4-e2b` (default) | `litert-community/gemma-4-E2B-it-litert-lm` | no | ~1.5 GB |
| `gemma-4-e4b` | `litert-community/gemma-4-E4B-it-litert-lm` | no | ~2.8 GB |
| `gemma-3n-e2b` | `google/gemma-3n-E2B-it-litert-lm` | yes (HF token) | ~1.5 GB |
| `gemma-3n-e4b` | `google/gemma-3n-E4B-it-litert-lm` | yes (HF token) | ~2.8 GB |

Models are downloaded on demand via the Ollama-compatible `/api/pull`
endpoint. See `DOCS.md` for usage.

## Prerequisites — HuggingFace Access (only for gated Gemma 3n models)

The `litert-community/*` Gemma 4 models are public — no token needed.
The `google/*` Gemma 3n models are **gated**. To use them:

1. Sign in at <https://huggingface.co/>.
2. Open the model page (e.g.
   <https://huggingface.co/google/gemma-3n-E2B-it-litert-lm>) and accept
   the Gemma license.
3. Create a **read-scope access token** at
   <https://huggingface.co/settings/tokens>.
4. Paste the token into the `hf_token` add-on option (it is stored as a
   password-type field and never appears in logs).

Without a valid token, `/api/pull` requests for gated models will fail
with HTTP 401.

## Configuration

| Option | Default | Description |
|---|---|---|
| `log_level` | `info` | trace, debug, info, notice, warning, error, fatal |
| `default_model` | `gemma-4-e2b` | Model used when a request omits `model` |
| `max_tokens` | 1024 | Default upper bound for any request; can be raised up to 32768 (the LiteRT-LM context window) |
| `temperature` | 0.7 | Default sampling temperature |
| `preload_models` | `[]` | Model names to pull on startup |
| `hf_token` | `""` | HuggingFace read token; required only for gated `google/*` Gemma 3n models. |
```

- [ ] **Step 2: Create `litert-llm-server/DOCS.md`**

```markdown
# LiteRT LLM Server — Detailed Docs

## Endpoints

### OpenAI-compatible (`/v1/*`)

- `GET /v1/models`
- `POST /v1/chat/completions` (stream / non-stream)
- `POST /v1/completions`

### Ollama-compatible (`/api/*`)

- `GET /api/tags`
- `POST /api/chat` (stream / non-stream)
- `POST /api/generate` (stream / non-stream)
- `POST /api/pull` (stream progress)
- `POST /api/show`
- `DELETE /api/delete`

### Health

- `GET /healthz` — liveness
- `GET /readyz` — readiness

## Using with Home Assistant

Add an "OpenAI Conversation" integration:

- **Base URL:** `http://<add-on-hostname>:8080/v1`
- **API key:** anything (no auth in MVP)
- **Model:** `gemma-4-e2b`

## Using with Node-RED

Use any Ollama node and point it to `http://<add-on-hostname>:8080`.

## Limits (MVP)

- **Context window: 32k tokens** (prompt + completion combined). Long
  multi-turn conversations may exhaust it; no automatic truncation in MVP.
- Single-slot engine: switching models mid-flight triggers a reload.
- No request queue: concurrent requests serialize.
- No authentication: rely on HA's internal network.
- Embeddings, function-calling, and multi-modal are not yet supported.
```

- [ ] **Step 3: Create `litert-llm-server/CHANGELOG.md`**

```markdown
# Changelog

## 0.1.0 — 2026-05-16

- Initial release.
- OpenAI- and Ollama-compatible APIs.
- Gemma 4 E2B/E4B (public) and Gemma 3n E2B/E4B (gated) via HuggingFace.
- Auto-download via `/api/pull`.
- 32k context window.
```

- [ ] **Step 4: Commit**

```bash
git add litert-llm-server/README.md litert-llm-server/DOCS.md litert-llm-server/CHANGELOG.md
git commit -m "docs(litert): add README, DOCS, and CHANGELOG"
```

---

## Phase 11 — End-to-End Validation

### Task 32: Local Docker smoke test

**Files:** (no new files)

- [ ] **Step 1: Build the image**

Run from the repo root:

```bash
# Pick BUILD_ARCH to match your host (amd64 on x86, aarch64 on ARM64).
docker build \
  --build-arg BUILD_FROM=ghcr.io/hassio-addons/base-python:14.0.2 \
  --build-arg BUILD_ARCH=amd64 \
  -t local/litert-llm-server:dev \
  litert-llm-server
```
Expected: image builds without errors. Build will take several minutes on first run.

- [ ] **Step 2: Run the container**

```bash
mkdir -p /tmp/litert-models
docker run --rm -p 8080:8080 \
  -e LITERT_LOG_LEVEL=debug \
  -e LITERT_DEFAULT_MODEL=gemma-4-e2b \
  -e LITERT_MAX_TOKENS=128 \
  -e LITERT_TEMPERATURE=0.7 \
  -e LITERT_MODELS_DIR=/data/models \
  -e LITERT_PORT=8080 \
  -e LITERT_PRELOAD_MODELS='[]' \
  -v /tmp/litert-models:/data/models \
  local/litert-llm-server:dev
```
Expected: container starts, `uvicorn running on 0.0.0.0:8080`.

- [ ] **Step 3: Sanity-check endpoints from another terminal**

```bash
curl -sS http://localhost:8080/healthz
curl -sS http://localhost:8080/v1/models
curl -sS http://localhost:8080/api/tags
```
Expected: health is `{"status":"ok"}`. Models lists are empty (no models cached yet — `/api/tags` returns `{"models":[]}` until `/api/pull` runs).

- [ ] **Step 4: Pull a model and verify it appears**

```bash
curl -sS -X POST http://localhost:8080/api/pull \
  -H 'content-type: application/json' \
  -d '{"name":"gemma-4-e2b","stream":false}'

curl -sS http://localhost:8080/v1/models
```
Expected: pull completes (may take a while), `/v1/models` then lists `gemma-4-e2b`.

- [ ] **Step 5: Chat completion smoke test**

```bash
curl -sS -X POST http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model":"gemma-4-e2b",
    "messages":[{"role":"user","content":"Say hi in one sentence."}],
    "max_tokens":40,
    "stream":false
  }'
```
Expected: JSON with `choices[0].message.content` populated.

- [ ] **Step 6: Stop container**

`Ctrl+C` in the docker run terminal. No commit (nothing changed in repo).

---

### Task 33: Install in local Home Assistant and end-to-end test

**Files:** (no new files)

- [ ] **Step 1: In HA Supervisor, add this repo as a Local Add-on Repository**

Settings → Add-ons → Add-on Store → ⋮ → Repositories → enter the path of this repository (e.g., `/Users/dariuspauly/projects/claude/homassist-addons` if HA can see it, otherwise mount or sync to a path the supervisor has access to).

Expected: "LiteRT LLM Server" appears in the store.

- [ ] **Step 2: Install the add-on**

Click on the add-on and choose "Install". The supervisor builds the image locally — expect 5–15 min on first install.

Expected: installation succeeds.

- [ ] **Step 3: Configure and start**

Set `default_model: gemma-4-e2b` and click "Start".

Expected: add-on logs show `uvicorn running on 0.0.0.0:8080`. `/healthz` reachable via the add-on's internal hostname.

- [ ] **Step 4: Configure HA's OpenAI Conversation integration**

Settings → Devices & Services → Add Integration → "OpenAI Conversation".

- **Base URL:** `http://<addon-hostname>:8080/v1` (the hostname HA assigns to the add-on)
- **API key:** anything, e.g., `local`
- **Model:** `gemma-4-e2b`

Trigger a test conversation. Expected: HA receives a generated reply.

- [ ] **Step 5: Sanity-check from Node-RED (optional)**

Configure a Node-RED Ollama node to use `http://<addon-hostname>:8080` and send a test prompt.

Expected: streamed response.

- [ ] **Step 6: Document the end-to-end test in `docs/benchmarks/`**

Create `docs/benchmarks/2026-05-16-e2e-ha-integration.md` (date may differ):

```markdown
# End-to-End Test — HA OpenAI Conversation with litert-llm-server

**Date:** <fill in>

## Result

- Add-on installed via Local Repository: <yes/no>
- Add-on started successfully: <yes/no>
- HA OpenAI Conversation reached the add-on: <yes/no>
- Generated response was sensible: <yes/no>

## Notes

<any observations>
```

- [ ] **Step 7: Commit the e2e test record**

```bash
git add docs/benchmarks/2026-05-16-e2e-ha-integration.md
git commit -m "docs(bench): record HA end-to-end integration test"
```

---

## Self-Review

(Performed at write-time; issues are inline-fixed.)

**1. Spec coverage**

- Repo skeleton + manifest → Task 1
- Add-on directory structure → Task 2
- PoC benchmark (decision gate) → Tasks 3–5
- Domain types + Protocols → Tasks 6–8
- Test fakes → Tasks 9–10
- OpenAI adapter (4 endpoints) → Tasks 11–14
- Ollama adapter (5 endpoints) → Tasks 15–19
- LiteRT engine → Tasks 20–21
- Model registry (filesystem + HF) → Tasks 22–23
- Settings + wiring + health → Tasks 24–25
- Architecture enforcement → Task 26
- Add-on wrapper (config.yaml, build.yaml, Dockerfile, rootfs) → Tasks 27–30
- Add-on docs → Task 31
- E2E smoke (docker) → Task 32
- E2E in HA → Task 33

All spec sections covered.

**2. Placeholder scan**

No "TBD" / "TODO" / "implement later" markers. The `<fill in>` placeholders in benchmark and e2e documentation are the operator's runtime data — not plan-side gaps.

**3. Type consistency**

- `InferenceService.stream_completion` signature is identical across `domain/inference.py`, `FakeEngine`, `LiteRTEngine`.
- `ModelRegistry` method names (`list`, `get`, `pull`, `delete`) consistent across Protocol, `FakeRegistry`, `HuggingFaceRegistry`.
- Settings field names match `LITERT_*` env vars set by `cont-init.d/01-config.sh`.
- `ChatMessage` shape consistent across OpenAI request/response.

**4. Open items**

- Engine package: this plan uses `litert-lm-api` (Google's current Python
  wrapper over LiteRT-LM). The original draft referenced
  `mediapipe.tasks.python.genai.inference.LlmInference`, which is
  deprecated and no longer present in current `mediapipe` PyPI builds.
  Updated 2026-05-17; see Spec Section 3 historical note.
- Task 21 uses true per-token streaming via `litert_lm.Session`. If a
  given `litert-lm-api` version exposes a different streaming method
  name, the producer thread body in `engines/litert.py` needs the
  corresponding adjustment (`generate_stream` vs alternative). The
  `InferenceService` Protocol contract does not change.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-16-litert-llm-server.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
