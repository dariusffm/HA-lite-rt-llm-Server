"""PoC benchmark: measure LiteRT-LM Gemma 2B sustained decode token rate.

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

MODEL_REPO = "google/gemma-2-2b-it-tflite"
MODEL_FILE = "gemma-2-2b-it-q8.task"
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
