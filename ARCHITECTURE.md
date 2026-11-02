# Architecture

How FluxLLM is put together: the module map, the request lifecycle, the
scheduler/engine contract, cache semantics, and the design decisions —
including what is deliberately out of scope.

## The one-paragraph version

A single async engine coroutine owns all model work. Each scheduler
iteration it asks the scheduler for a *plan* (which queued requests to
prefill, which admitted requests to decode), executes that plan against a
`ModelExecutor` in a worker thread, streams every sampled token as an event,
and advances each sequence's KV accounting by exactly one block-managed
slot. Capacity is checked before admission; running out of pool mid-flight
rolls one step back and preempts (the request re-queues and its KV state is
recomputed by re-prefilling prompt + generated tokens, vLLM-style). API and
demo layers are thin shells over `submit/stream/stats/metrics`.

## Module map

```
fluxllm/
├── config/settings.py      EngineSettings (env prefix FLUXLLM_, pydantic-settings)
├── engine/
│   ├── engine.py           FluxEngine: loop, submission, streaming, preemption
│   ├── lifecycle.py        RequestState enum + transition validation
│   ├── request.py          GenerationRequest + RequestTimings (TTFT etc.)
│   ├── detokenizer.py      incremental stable text deltas for SSE
│   └── tokens.py           deterministic simulation tokenizer (labeled)
├── executor/
│   ├── base.py             ModelExecutor protocol + StepOutput/SequenceState
│   ├── hf_executor.py      HF causal LM, per-sequence DynamicCache KV
│   ├── deterministic.py    numpy simulation executor for tests/demos
│   └── device.py           device/dtype resolution
├── cache/
│   └── block_manager.py    BlockManager (logical block pool) + PrefixCache
├── scheduler/
│   ├── scheduler.py        queue + running set + plan()/advance/preempt
│   └── policy.py           admission checks (batch size, tokens, KV headroom)
├── sampling/               SamplingParams + numpy sampler
├── metrics/                EngineMetrics (Prometheus) + global singleton
└── api/                    FastAPI app/routes/schemas, SSE, /demo mount
```

## Request lifecycle

```
QUEUED ──admit──> PREFILL ──first token──> DECODING ──eos/stop/length──> COMPLETED
   │                  │                                        ├──> CANCELLED
   └──no capacity────>├──stalled (can never fit)──────────────> FAILED
                      └──pool exhausted mid-decode──> preempt──> back to QUEUED
                                   (preempted=True, KV recomputed on re-admission)
```

Every transition is validated by `lifecycle.py`; every terminal state
releases scheduler entries, executor KV state, and pool blocks exactly once
(verified with pool-invariant assertions in tests).

## Engine loop (per iteration)

1. `scheduler.plan()` — strict FIFO: the head of the queue is checked
   against batch size, per-iteration token budget, and `lookahead_tokens=8`
   block headroom (`1` step headroom for preempted requests). The first
   request that does not fit blocks *everyone behind it* (strict FIFO is a
   documented policy, not an accident). Admitted requests are pre-allocated
   prompt blocks immediately.
2. Prefills — newly admitted requests run in a worker thread
   (`asyncio.to_thread`). When more than one is ready, `prefill_batch` packs
   them into one left-padded model forward; rows that share a cached prompt
   prefix reuse its KV instead of recomputing it. The first token is sampled
   and streamed for each request. TTFT is recorded here (submission → first
   streamed token).
3. Decodes — for every decoding sequence: sample from the pending logits,
   stream the token, reserve one KV slot (`block_manager.extend`), and feed
   the token through one decode forward. Ready sequences share a single
   `decode_batch` forward. Terminal tokens (EOS / stop ids) append and
   finish without another forward.
4. If `extend` cannot fit the next slot, the engine rolls the token back
   (`unextend`), releases the sequence, re-queues it as preempted, and
   emits a preemption metric. A head request that can *never* fit (bigger
   than the whole pool) is failed with a clear error so the queue advances.
5. Telemetry (`on_iteration` gauges/counters) is recorded *after* committed
   work executes — a metrics failure can never strand an admission.

## Cache semantics and honest-scope notes

`BlockManager` is **pure bookkeeping** — it never touches tensors:

- fixed-size blocks (`block_size`, default 16) form a bounded pool
  (`num_blocks`); sequences hold refcounted block tables;
- `allocate` (admission), `extend`/`unextend` (decode-step accounting),
  `can_allocate`/`can_extend` (pre-admission questions), `release`
  (completion) are the full API;
- blocks are refcounted because multiple sequences can own the *same*
  block through the prefix cache;
- `check_invariants()` (used in tests) asserts refs / content / block-count
  consistency — the accounting contract the scheduler depends on.

`PrefixCache` gives identity to full blocks via chains of per-block content
hashes (zlib.crc32 per block, cumulative chain keys). Reuse is implemented
by refcount sharing: a matching incoming request is registered against the
same block instead of renting a fresh one.

**Two layers of prefix reuse.** The block manager shares *capacity*:
two sequences that match on a full block rent the same block. The HF
executor separately shares *compute*: it keeps a small bounded LRU of
cloned KV snapshots keyed by prompt token ids. A later prefill with an
exact match returns the stored next-token logits (zero forwards), a shorter
cached prefix forwards only the uncached tail, and a prompt that is a
prefix of a cached longer prompt crops the cached KV and forwards one token.
Snapshots are cloned, never shared, because decoding mutates a cache in
place. This is executor-level compute reuse, observable via
`fluxllm_prefix_compute_*`; it is orthogonal to the block pool's accounting.

**What this is NOT:** paged attention. The HF executor still keeps
per-sequence KV tensors (transformers `DynamicCache`); batched forwards
gather rows into a left-padded batch and slice them back out, and two
sequences sharing a block still own their own KV tensors. Batched decode is
therefore model-forward batching, not a fused kernel reading pages across
sequences. True PagedAttention-style sharing would require a paged KV
allocator plus a batched attention kernel — deliberately future work. Every
metric name says "accounting events" or "compute reuse" and the docs never
borrow kernel-level claims.

## Sampling

Pure numpy: temperature scaling, top-k, top-p; `temperature=0` means greedy
argmax; per-request seeds produce reproducible streams. Tests pin exact
token sequences on fixed seeds, so sampling regressions are caught, not
noticed.

## Executors

`ModelExecutor` is intentionally minimal: `prefill(request_id, tokens)` →
`StepOutput` (next-token logits + opaque `SequenceState`), `decode(state,
last_token)`, `release(request_id)`. The optional batch methods
`prefill_batch` / `decode_batch` have the same per-row contract, so the
engine can fall back to the sequential path without changing semantics. Two
implementations:

- `HFModelExecutor` — real PyTorch/transformers execution with per-sequence
  `DynamicCache`, left-padded batched forwards, and a bounded prefix-KV
  snapshot store for compute reuse; runs on CPU or CUDA.
- `DeterministicExecutor` — dependency-free numpy simulation whose logits
  pattern is fixed arithmetic, always clearly labeled as simulation (it
  powers unit tests, the CI fast lane, and demo/screen modes that must not
  present fake model output as real). It mirrors the batch and prefix-reuse
  interfaces so the fast lane exercises the same engine paths.

## Metrics

One `EngineMetrics` instance (process-wide `global_metrics`) with the
`fluxllm_` prefix exposes:

- counters: requests received/completed/cancelled/failed, preemptions,
  scheduler iterations, prompt/generated tokens, prefix-cache
  hit/miss/reused-tokens, batched model forwards, prefix-compute reuses and
  reused tokens;
- gauges: queue depth, running requests, batch size, KV blocks
  used/free/utilization, prefix hit rate;
- histograms: TTFT, inter-token latency, prefill latency, e2e latency.

Design rule learned the hard way: counters are fed **per event**
(allocation-time for the prefix cache) — never by snapshot deltas, because
per-engine snapshots reset while the singleton persists, and a negative
`Counter.inc` is an exception that can strand committed work.

## Concurrency model

- the engine loop is one asyncio task on the app's loop;
- model forwards run in worker threads (`asyncio.to_thread`), so the loop
  stays responsive to submissions and cancellations;
- per-request sampling and KV bookkeeping stay on the loop thread (no
  locks needed; the executor is the only shared mutable object and each
  call owns exactly one sequence's state);
- client disconnects surface as `asyncio.CancelledError` in the SSE
  generator, which calls `handle.cancel()` — cancellation lands at the
  next engine step boundary, and *stats* confirm full resource release.

## Testing strategy

- **Fast lane** (`pytest tests`, default): deterministic executor only, no
  model downloads, no GPU — runs in a few seconds and covers the engine,
  scheduler, cache (including invariants), batched-path equivalence, prefix
  reuse bookkeeping, sampling, metrics, API, SSE race behavior, and
  incremental detokenization (including partial UTF-8).
- **Integration lane** (`FLUXLLM_INTEGRATION=1`, weekly in CI): loads the
  real Qwen model and checks real prefill/decode, batched-vs-sequential
  equivalence, prefix-reuse logits against a cold prefill, streaming, and
  resource release.
- **Linux parity**: the CI-relevant paths were also verified in a local
  `python:3.11-slim` container (this caught a platform-only mypy failure
  and a streaming race that never reproduced on Windows).
