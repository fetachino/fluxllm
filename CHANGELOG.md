# Changelog

All notable changes to FluxLLM are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); every "Added"
entry maps to a real commit, and every performance claim maps to a
committed measurement artifact — never to a projection.

## [1.0.0] - 2026-10-31

First complete release of the engine: continuous batching, block-based
KV management, streaming, metrics, OpenAI-compatible API, demo UI,
benchmarks, and CI. Everything listed below is implemented in this
repository (no vLLM/TGI/llama.cpp underneath; PyTorch +
`transformers` only).

### Added

- **Engine core** — single async engine loop with per-iteration planning,
  prefill/decode phases, vLLM-style preemption with KV recompute, and a
  stall-fail path for requests that can never fit
  (`22d1bc2`, `94cd66e`). Request lifecycle is a validated state machine
  (`0a0b41c`).
- **Block-based KV cache management** — refcounted logical block pool with
  pre-admission capacity checks and pool-invariant assertions
  (`9b5c11e`).
- **Logical prefix cache** — content-hash chains (crc32 per full block)
  with refcount sharing; accounting-level reuse only, explicitly not
  PagedAttention (`030ef55`).
- **Sampling** — temperature / top-k / top-p / greedy, per-request seeds,
  exact-sequence determinism tests (`94cd66e`).
- **Metrics** — `fluxllm_`-prefixed Prometheus counters/gauges/histograms
  (TTFT, ITL, e2e, KV utilization, queue depth, prefix cache), fed **per
  event** after committed work — never via per-iteration snapshot deltas;
  the cross-engine delta bug class is pinned by a regression test
  (`f187685`, `030ef55`).
- **API** — OpenAI-compatible `/v1/completions` and
  `/v1/chat/completions` (sync + SSE streaming), `/health/live`,
  `/health/ready`, `/stats`, `/metrics`; disconnect-driven server-side
  cancellation (`d093a99`).
- **Demo UI** — `/demo` page with streaming, TTFT/rate/finish
  chips, live engine gauges, and a 4-way burst view showing arrivals
  joining mid-decode (`3a6aa31`). Screenshots in `docs/images/` are real
  captured runs (8 admissions, 0 preemptions, staggered TTFTs
  0.40–2.20s).
- **Benchmarks** — `benchmarks/benchmark_throughput.py` compares the
  engine against a sequential `transformers` baseline at concurrency
  1/2/4/8 with committed artifacts and full environment snapshots
  (`0ce6921`). Measured on the dev CPU machine
  (`benchmarks/results/run_20261031T051606Z.json`): sequential baseline
  6.64 tok/s; FluxLLM 6.45 / 6.45 / 6.50 / 6.68 tok/s at c=1/2/4/8;
  TTFT p50 0.37 → 2.71s from c=1 to c=8; 0 preemptions. Aggregate
  throughput is honest about being flat — per-sequence decode is the
  documented next lever (#1).
- **Documentation** — README, ARCHITECTURE, BENCHMARKS, ROADMAP,
  SECURITY, deployment guide (Docker/Compose/Coolify), interview guide,
  resume notes (`63126d2`).

### Fixed

- **Streaming race under Linux** — SSE generators re-decoded the
  request's shared `generated_token_ids` per event; when the engine
  sampled ahead of consumption, streamed chunk counts varied (4 on
  Windows, 2–3 on Linux). Pieces now diff from event-locally accumulated
  id streams; verified with 15 consecutive green runs in a
  `python:3.11-slim` container matching CI (`fef8f5e`).
- **Executor typing across environments** — transformers 5.x wraps
  `Module.to` and its stubs resolve inconsistently with/without torch
  installed; routed through an explicit `Any` cast so both worlds are
  mypy-clean (`fef8f5e`).
- **Docker stage naming** — API stage renamed to `api` to match CI's
  `--target api` and compose; both targets build and the API image
  passes a real smoke run (`df05098`).

### Infrastructure

- **CI fast lane** — ruff, mypy (strict-ish), pytest with coverage ≥ 70%,
  and Docker builds of both targets on every push; no model downloads,
  no GPU. Real-model integration runs on a weekly schedule and on
  demand (`9142f19`, refined by later commits).
- **Docker** — multi-stage CPU-torch image (non-root user, healthcheck)
  and hardened compose (read-only rootfs, tmpfs home, `hf-cache`
  volume) (`9142f19`, `df05098`).

### Known limitations (honest, by design)

- Decode forwards are per sequence; aggregate throughput does not scale
  with concurrency yet — tracked in #1 with measured evidence.
- Prefix cache shares pool capacity, not executor-side KV tensors — see
  #2.
- No authentication/TLS/rate-limiting layer; exposed deployments need a
  reverse proxy (SECURITY.md).
- Benchmarks are CPU-only, environment-specific, and never projected.

[1.0.0]: https://github.com/fetachino/fluxllm/releases/tag/v1.0.0
