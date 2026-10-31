# Roadmap

Ordered by what would make the biggest honest difference. Items move to
"Benchmarked" only after they ship with real measurements in
[BENCHMARKS.md](BENCHMARKS.md); nothing is claimed before it is measured.

## In progress / next

- **Batched decode execution** — group all decoding sequences into one
  forward where the executor supports it (padding + per-sequence KV with
  position masking, kernel-level batching). This is the single biggest
  throughput lever missing from FluxLLM: current per-sequence decode price
  is documented in BENCHMARKS.md (flat aggregate throughput at
  concurrency 1–8 on CPU).
- **Prefix-cache compute reuse** — pair the existing accounting-level
  reuse with executor-level shared prefill (at least skip re-prefilling
  shared full blocks by extending from the shared KV state where the
  executor supports it).
- **Streaming token = one chunk, not detokenized text diff** — append
  incremental decode (prefix-consistent incremental detokenizer) so
  partial multibyte/merge situations stream cleanly.

## Planned

- **Continuous-flow tuning** — per-iteration prefill budget rebalancing
  (priority tokens to prefill vs decode), preemption heuristics that
  consider time-remaining rather than strict FIFO only.
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

- Claiming vLLM PagedAttention parity ever. The prefix cache is
  accounting-level and documented as such.
- Fabricated performance numbers, screenshots, or deployments. Nothing
  enters the docs before it is measured on the recorded environment.
