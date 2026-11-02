# Roadmap

Ordered by what would make the biggest honest difference. Items move to
"Benchmarked" only after they ship with real measurements in
[BENCHMARKS.md](BENCHMARKS.md); nothing is claimed before it is measured.

## In progress / next

- **Paged-attention kernel** — the current batched path gathers per-sequence
  KV into a left-padded batch and slices it back out every step. That is a
  real gain (2.1× at concurrency 8 on CPU) but it plateaus beyond two rows
  because gather/scatter and padding overhead grow with batch size. A fused
  kernel reading pages across sequences is the next throughput lever, and it
  will not be claimed before it is measured.
- **Prefix-reuse throughput benchmark** — correctness and counters are
  verified; measure the wall-clock win on repeated/shared-prefix traffic so
  the claim is empirical rather than structural.
- **Continuous-flow tuning** — per-iteration prefill budget rebalancing
  (priority tokens to prefill vs decode), preemption heuristics that
  consider time-remaining rather than strict FIFO only.

## Planned

- **More stop conditions** — stop strings (matching on decoded suffix),
  min-new-tokens, logit bias.
- **Multi-model serve** — one process, several engines, per-model queues
  and metrics (low priority; single-model correctness first).
- **CUDA path** — GPU device support beyond `device=auto` resolution,
  benchmarked on a real GPU before any numbers are claimed.
- **JSON/schema-constrained decoding** — mask logits with a schema DFA.

## Done

- ✅ Request lifecycle state machine, engine loop, streaming, preemption
  + KV-recompute, stall-fail path.
- ✅ Block-based KV management with refcounts + invariants; logical
  prefix cache with honest accounting-level semantics.
- ✅ Batched decode and prefill execution (`prefill_batch` /
  `decode_batch`), with measured throughput in BENCHMARKS.md.
- ✅ Executor-level prefix compute reuse: exact, partial-prefix, and
  covering-prefix reuse with bounded LRU snapshots, metrics, and real-model
  equivalence tests.
- ✅ Incremental detokenizer for SSE: stable text deltas, held-back partial
  multi-byte characters, finalize-on-stop.
- ✅ OpenAI-compatible API including SSE streaming and
  disconnect-cancellation; race-free SSE piece diffing (fixed under Linux).
- ✅ Prometheus metrics fed per event; /stats live snapshot.
- ✅ Reproducible benchmarks with measured CPU results.
- ✅ Demo UI with live engine gauges + burst view; real screenshots.
- ✅ CI: ruff/mypy/pytest-coverage fast lane + Docker builds;
  `model-integration.yml` weekly real-model lane.
- ✅ Dockerfile (non-root, CPU torch, healthcheck), compose
  (read-only rootfs, tmpfs, hf-cache volume).

## Non-goals

- Claiming vLLM PagedAttention parity ever. KV tensors are per sequence;
  batching gathers rows into a padded batch and the docs say so.
- Fabricated performance numbers, screenshots, or deployments. Nothing
  enters the docs before it is measured on the recorded environment.
