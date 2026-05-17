# Benchmark — LiteRT-LM Gemma-4-E2B (Smoke Test)

**Date:** 2026-05-17
**Host:** macOS-arm64 (Apple Silicon dev machine — **NOT** the amd64 NUC target)
**Model:** `litert-community/gemma-4-E2B-it-litert-lm` / `gemma-4-E2B-it.litertlm` (int4, public)
**Script:** `litert-llm-server/app/scripts/bench_gemma2b.py`
**Engine:** `litert_lm.Benchmark` with `Backend.CPU`
**Prefill tokens:** 256
**Decode tokens:** 128

## Purpose

This is a **smoke test**, not the spec-mandated acceptance benchmark. It
runs on the developer's Apple Silicon machine rather than the amd64
NUC/server that is the actual deployment target. The goal here is to
verify the entire LiteRT-LM stack works end-to-end (library imports,
model download, engine initialization, decode loop, metric collection)
before committing to building the rest of the application against it.

The proper amd64-target acceptance benchmark remains a prerequisite
before the add-on is considered production-ready; it will live in this
directory under a `2026-05-XX-litert-gemma4-e2b-amd64.md` filename when
it runs.

## Result

| Metric | Value |
|---|---|
| init_time | 2.95 s |
| time_to_first_token | 1.41 s |
| prefill_tokens_per_second | 185.32 |
| **last_decode_tokens_per_second** | **36.53** |
| Acceptance threshold (decode) | ≥ 5.0 tok/s |
| **Decision** | **PASS** |

## Interpretation

- The decode rate exceeds the threshold by ~7×. Even if the amd64 NUC is
  significantly slower (different SIMD path, no Apple ML accelerator), a
  ~5× degradation would still leave it above acceptance.
- Token-to-first-token (1.4 s) is reasonable for a 2B-effective-parameter
  model with a 256-token prefill on CPU.
- Engine init in under 3 s suggests model loading is not a startup
  bottleneck for typical sessions.

## Decision

**Proceed with LiteRT-LM as the inference engine.** No engine swap to
`llama-cpp-python` is required at this stage. The Phase 1 decision gate
is cleared on the strength of the smoke result; final acceptance
remains contingent on the amd64 benchmark.

## Caveats

- This run uses Apple Silicon's optimized BLAS/ML paths. amd64 will use
  XNNPACK or AVX2/AVX-512 kernels; absolute numbers will differ.
- The first-token latency includes JIT compilation in some backends.
  Subsequent inferences within the same engine instance may be faster.
- This test downloaded the public `litert-community/*` build. The gated
  `google/*-litert-lm` Gemma 3n models were not exercised here.

## Reproduce

```bash
cd litert-llm-server/app
uv sync --extra dev
uv run python scripts/bench_gemma2b.py
```

Public model (`litert-community/gemma-4-E2B-it-litert-lm`): no HF token
required.
