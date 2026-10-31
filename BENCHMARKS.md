# Benchmarks

Reproducible throughput measurements for FluxLLM against a plain
`transformers` sequential baseline. Every number here comes from an actual
run whose raw artifacts are committed under `benchmarks/results/` — nothing
in this document is projected, extrapolated, or fabricated. If a number is
missing, that measurement has not been taken.

## Environment

Measured on the following machine (from the run's `env` snapshot,
`benchmarks/results/run_20261031T051606Z.json`):

| field | value |
| --- | --- |
| date (UTC) | 2026-10-31T05:16:06Z |
| platform | Windows-10, AMD64 (laptop, 8-core/16-thread Ryzen-class CPU) |
| logical CPUs / torch threads | 16 / 8 |
| Python / torch / transformers | 3.11.9 / 2.15.0+cpu / 5.22.1 |
| model | `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0, ~1 GB) |
| precision / device | float32 / cpu (no CUDA on this machine) |
| code revision | `030ef55e8f60b963978bac9d910dfb776a2b6e2f` |

## Methodology

Both modes decode the **same fixed prompt set** (8 varied short English
prompts, ~10–15 tokens each) with **greedy decoding** (`temperature=0`,
`top_k=1`, 32 `max_new_tokens` per request — 256 tokens total per run), so
the only difference between runs is how requests are scheduled:

- **Sequential baseline** — `transformers` `AutoModelForCausalLM.generate`,
  one request after another (the naive way to serve a fixed queue).
- **FluxLLM** — the continuous-batching engine at concurrency
  *c* ∈ {1, 2, 4, 8}: the first *c* requests are admitted together, and a
  slot frees the next queued request starts immediately (no batch drain).
  Each request streams through the engine's token events; TTFT and
  end-to-end latency come from the request's own timing checkpoints.
- Each mode runs one unmeasured warm-up request. Pool sizing (96 blocks ×
  16 tokens) was generous enough that **zero preemptions** occurred in any
  run, so these numbers measure scheduling, not preemption/recompute churn.

## Measured results

8 requests × 32 tokens each, per mode (`benchmarks/results/`):

| mode | concurrency | wall (s) | agg. throughput (tok/s) | TTFT p50 (s) | TTFT p95 (s) | e2e p50 (s) | preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential transformers | 1 | 38.6 | 6.64 | — | — | 4.78 | — |
| fluxllm | 1 | 39.7 | 6.45 | 0.37 | 0.59 | 4.93 | 0 |
| fluxllm | 2 | 39.7 | 6.45 | 0.86 | 1.09 | 9.85 | 0 |
| fluxllm | 4 | 39.4 | 6.50 | 1.35 | 2.09 | 19.72 | 0 |
| fluxllm | 8 | 38.3 | 6.68 | 2.71 | 4.26 | 38.31 | 0 |

![FluxLLM throughput vs concurrency](docs/images/benchmark_throughput.png)
![TTFT by concurrency](docs/images/benchmark_ttft.png)

## Interpretation (honest read)

- **c = 1 parity anchors the engine.** FluxLLM at concurrency 1 (6.45
  tok/s, e2e p50 4.93 s) matches the sequential `transformers` baseline
  (6.64 tok/s, 4.78 s) within ~3%: the scheduler, sampling, streaming, and
  bookkeeping layers add almost nothing to a single stream's cost.
- **Aggregate throughput does not scale with concurrency on this
  hardware — and that is the documented architecture, not noise.** Decode
  forwards in this executor are *per sequence* (no batched attention
  kernel; see `ARCHITECTURE.md` scope notes). Each forward already
  saturates torch's 8 CPU threads, and one scheduler iteration awaits all
  of its per-sequence decodes, so serving 8 sequences concurrently overlaps
  queueing/bookkeeping but performs the same total matrix math. Kernel-
  level batched decoding remains future work (ROADMAP), and until it
  exists these flat-throughput curves are the truthful expectation.
- **Concurrency trades per-request latency for arrival-_handling.** At
  c = 8 every request still progresses continuously (each request
  averages ~0.84 tok/s → 32 tokens in ~38 s), but TTFT rises with queue
  depth (0.37 s → 2.71 s p50) because each iteration serializes
  *c* forwards before the next token streams. No request was preempted or
  starved; all 8 finished with `reason=length`.
- **What continuous batching demonstrably delivered here:** arrivals are
  admitted and prefilled while earlier requests are mid-decode (no
  wait-for-batch-drain), queue depth, batch size, and KV utilization are
  exposed live via `/stats` and Prometheus gauges, and completion/cancel
  paths free blocks immediately. The continuous-batching scheduling
  property is pinned down in `tests/unit/test_engine.py`
  (`test_later_arrivals_share_engine_iterations`).

## What is *not* benchmarked (yet)

- Anything GPU/CUDA; this development machine has none, so no GPU numbers
  exist anywhere in this repo.
- Prefix-cache hit-rates under real repeated-prompt traffic: the 8 prompts
  are all distinct, so the measurement exercises the accounting paths
  (`/metrics` `fluxllm_prefix_cache_*`) but not reuse gains. Reuse is
  accounting-level (fewer blocks rented), not kernel sharing.
- Batched attention kernels, CUDA graphs, or speculative decoding: not
  implemented, never claimed.

## Reproduce

```bash
# ~4 minutes on the machine above (downloads ~1 GB of weights on first run)
python -m benchmarks.benchmark_throughput --concurrency 1 2 4 8 --max-new-tokens 32 --warmup 1

# regenerate the PNG charts from the newest artifact
python scripts/make_charts.py
```

Artifacts land in `benchmarks/results/run_<UTC>.json` (with a full
environment snapshot and per-request timings) and
`summary_<UTC>.csv` (tight table matching the columns above). Include the
`commit_sha` from the JSON when reporting numbers elsewhere.
