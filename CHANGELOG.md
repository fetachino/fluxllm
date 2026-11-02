# Changelog

All notable changes to FluxLLM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); every entry maps
to a real change in this repository, and every performance claim maps to a
committed measurement artifact — never to a projection.

Commit hashes are intentionally not embedded in this public copy: the
history was rewritten to remove private career notes, so pre-rewrite hashes
no longer resolve. Entries are identifiable by their messages and by the
artifacts they produced.

## [Unreleased]

### Added

- **Batched decode and prefill execution** (`#1`) — ready sequences in a
  scheduler iteration are packed into one left-padded model forward per
  phase (`prefill_batch` / `decode_batch`) instead of one forward per
  sequence. Per-row KV is gathered from each sequence's `DynamicCache`,
  advanced with per-row attention masks and position ids, then sliced back
  out. The engine falls back to the sequential path per row on failure.
  `FLUXLLM_ENABLE_BATCHED_EXECUTION=false` restores the pre-batching path,
  and `fluxllm_batched_model_forwards` reports grouped forwards.
- **Executor-level prefix compute reuse** (`#2`) — a bounded LRU of cloned
  KV snapshots keyed by prompt token ids. An exact prompt match returns the
  stored next-token logits with zero forwards; a shorter cached prefix
  forwards only the uncached tail; a prompt that is a prefix of a cached
  longer prompt crops the cached KV and forwards one token. Snapshots are
  cloned, never shared, because decode mutates a cache in place.
  `fluxllm_prefix_compute_reuses_total` and
  `fluxllm_prefix_compute_reused_tokens_total` expose reuse, and real-model
  integration tests compare reused logits against a cold prefill.
- **Incremental detokenizer** (`#3`) — SSE responses now emit stable text
  deltas from `IncrementalDetokenizer`, which holds back trailing partial
  UTF-8 sequences (U+FFFD) until the next token completes them and flushes
  the remainder on `finalize()`. This removes replacement-character churn
  and duplicate/retracted text from streamed completions.
- **Benchmark harness** — `--compare-batched` measures batched and
  per-sequence decode in the same run; `--no-batched-decode` selects the
  pre-batching path; JSON payloads record the settings block, and CSV/JSON
  rows carry `batched_execution`. Charts group the two decode paths.

### Measured

- **Batched decode, CPU dev machine, 8 requests × 32 tokens**
  (`benchmarks/results/run_20261102T180750Z.json`): batched decode reaches
  **12.97 tok/s at concurrency 8 vs 6.18 tok/s per-sequence (2.10×)** and
  peaks at **15.67 tok/s at concurrency 2 (2.76× vs per-sequence)**. TTFT
  p50 at concurrency 8 drops from 2.69 s to 0.97 s. Throughput plateaus
  beyond two rows because the executor still gathers/pads/slices KV each
  step; a fused paged-attention kernel is future work. See
  [BENCHMARKS.md](BENCHMARKS.md) for the controlled comparison and the
  repeat run.

### Known limitations (honest, by design)

- Batched decode is model-forward batching, not a paged-attention kernel;
  it does not scale linearly with batch size.
- Prefix compute reuse is bounded (8 snapshots / 8192 prompt tokens by
  default) and uses cloned per-sequence KV, not paged KV sharing.
- No authentication/TLS/rate-limiting layer; exposed deployments need a
  reverse proxy (SECURITY.md).
- Benchmarks are CPU-only, environment-specific, and never projected.

## [1.0.0] - 2026-10-31

First complete release of the engine: continuous batching, block-based
KV management, streaming, metrics, OpenAI-compatible API, demo UI,
benchmarks, and CI. Everything listed below is implemented in this
repository (no vLLM/TGI/llama.cpp underneath; PyTorch +
`transformers` only).

### Added

- **Engine core** — single async engine loop with per-iteration planning,
  prefill/decode phases, vLLM-style preemption with KV recompute, and a
  stall-fail path for requests that can never fit. Request lifecycle is a
  validated state machine.
- **Block-based KV cache management** — refcounted logical block pool with
  pre-admission capacity checks and pool-invariant assertions.
- **Logical prefix cache** — content-hash chains (crc32 per full block)
  with refcount sharing; accounting-level capacity reuse only, explicitly
  not PagedAttention.
- **Sampling** — temperature / top-k / top-p / greedy, per-request seeds,
  exact-sequence determinism tests.
- **Metrics** — `fluxllm_`-prefixed Prometheus counters/gauges/histograms
  (TTFT, ITL, e2e, KV utilization, queue depth, prefix cache), fed **per
  event** after committed work — never via per-iteration snapshot deltas;
  the cross-engine delta bug class is pinned by a regression test.
- **API** — OpenAI-compatible `/v1/completions` and
  `/v1/chat/completions` (sync + SSE streaming), `/health/live`,
  `/health/ready`, `/stats`, `/metrics`; disconnect-driven server-side
  cancellation.
- **Demo UI** — `/demo` page with streaming, TTFT/rate/finish
  chips, live engine gauges, and a 4-way burst view showing arrivals
  joining mid-decode. Screenshots in `docs/images/` are real captured runs
  (8 admissions, 0 preemptions, staggered TTFTs 0.40–2.20s).
- **Benchmarks** — `benchmarks/benchmark_throughput.py` compares the
  engine against a sequential `transformers` baseline at concurrency
  1/2/4/8 with committed artifacts and full environment snapshots.
  Measured on the dev CPU machine
  (`benchmarks/results/run_20261031T051606Z.json`): sequential baseline
  6.64 tok/s; FluxLLM 6.45 / 6.45 / 6.50 / 6.68 tok/s at c=1/2/4/8;
  TTFT p50 0.37 → 2.71s from c=1 to c=8; 0 preemptions. Aggregate
  throughput was flat on the then-per-sequence decode path; the batched
  path supersedes it in Unreleased.
- **Documentation** — README, ARCHITECTURE, BENCHMARKS, ROADMAP,
  SECURITY, and deployment guide (Docker/Compose/Coolify).

### Fixed

- **Streaming race under Linux** — SSE generators re-decoded the
  request's shared `generated_token_ids` per event; when the engine
  sampled ahead of consumption, streamed chunk counts varied (4 on
  Windows, 2–3 on Linux). Pieces now diff from event-locally accumulated
  id streams; verified with 15 consecutive green runs in a
  `python:3.11-slim` container matching CI.
- **Executor typing across environments** — transformers 5.x wraps
  `Module.to` and its stubs resolve inconsistently with/without torch
  installed; routed through an explicit `Any` cast so both worlds are
  mypy-clean.
- **Docker stage naming** — API stage renamed to `api` to match CI's
  `--target api` and compose; both targets build and the API image
  passes a real smoke run.

### Infrastructure

- **CI fast lane** — ruff, mypy (strict-ish), pytest with coverage ≥ 70%,
  and Docker builds of both targets on every push; no model downloads,
  no GPU. Real-model integration runs on a weekly schedule and on
  demand.
- **Docker** — multi-stage CPU-torch image (non-root user, healthcheck)
  and hardened compose (read-only rootfs, tmpfs home, `hf-cache`
  volume).

### Known limitations at 1.0.0 (superseded by Unreleased)

- Decode forwards were per sequence; aggregate throughput did not scale
  with concurrency — addressed by batched execution above.
- Prefix cache shared pool capacity, not executor-side compute — addressed
  by executor prefix compute reuse above.
- No authentication/TLS/rate-limiting layer; exposed deployments need a
  reverse proxy (SECURITY.md).
- Benchmarks are CPU-only, environment-specific, and never projected.

[Unreleased]: https://github.com/fetachino/fluxllm/compare/v1.0.0...main
[1.0.0]: https://github.com/fetachino/fluxllm/releases/tag/v1.0.0
