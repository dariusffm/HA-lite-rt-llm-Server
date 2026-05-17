"""PoC benchmark: measure LiteRT Gemma 2B sustained token rate.

Acceptance criterion (from spec): >= 5 tok/s sustained over 100 generated
tokens on the target amd64 hardware. Below that, the engine choice must be
re-evaluated before continuing.

Run on the actual target NUC/server:
    cd litert-llm-server/app
    uv run python scripts/bench_gemma2b.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from mediapipe.tasks.python.genai import bundler  # noqa: F401  (sanity import)
from mediapipe.tasks.python.genai.inference import (
    LlmInference,
    LlmInferenceOptions,
)

MODEL_REPO = "google/gemma-2-2b-it-tflite"
MODEL_FILE = "gemma-2-2b-it-q8.task"
PROMPT = "Write a short paragraph about home automation."
NUM_TOKENS = 100
ACCEPTANCE_TOK_S = 5.0


def download_model(cache_dir: Path) -> Path:
    print(f"Downloading {MODEL_REPO}/{MODEL_FILE} ...")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def benchmark(model_path: Path) -> float:
    print(f"Loading model: {model_path}")
    options = LlmInferenceOptions(
        model_path=str(model_path),
        max_tokens=NUM_TOKENS + 32,
        random_seed=42,
        top_k=40,
        temperature=0.7,
    )
    llm = LlmInference.create_from_options(options)

    print(f"Generating {NUM_TOKENS} tokens for prompt: {PROMPT!r}")
    start = time.perf_counter()
    output = llm.generate_response(PROMPT)
    elapsed = time.perf_counter() - start

    # Heuristic: split output into "tokens" by whitespace for a coarse
    # tok/s. The MediaPipe API does not stream token-by-token in this
    # synchronous call; this is good enough for the gate decision.
    token_count = max(len(output.split()), 1)
    tok_per_s = token_count / elapsed
    print(f"Generated {token_count} tokens in {elapsed:.2f}s -> {tok_per_s:.2f} tok/s")
    return tok_per_s


def main() -> int:
    cache_dir = Path(os.environ.get("LITERT_BENCH_CACHE", "./.models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = download_model(cache_dir)
    tok_per_s = benchmark(model_path)

    print("\n=== RESULT ===")
    print(f"Sustained rate: {tok_per_s:.2f} tok/s (acceptance: >= {ACCEPTANCE_TOK_S} tok/s)")
    if tok_per_s < ACCEPTANCE_TOK_S:
        print("BELOW ACCEPTANCE THRESHOLD — stop and re-evaluate engine choice.")
        return 1
    print("Above threshold — proceed with LiteRT engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
