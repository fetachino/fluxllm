"""Scheduler admission policy.

The policy answers two questions each iteration: which queued requests can
be admitted, and whether running sequences may keep growing. Decisions are
pure functions of (request, resources) so they are unit-testable without an
engine.

FIFO order is strict: if the head of the queue cannot be admitted this
iteration (batch full, token budget exhausted, or insufficient KV blocks),
no later request is admitted either. This keeps queue order fair and makes
scheduler behavior predictable enough to test exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from fluxllm.engine.request import GenerationRequest


@dataclass(frozen=True)
class SchedulerLimits:
    """Static capacity limits the scheduler must respect."""

    max_batch_size: int = 8  # max concurrently running sequences
    max_num_batched_tokens: int = 512  # prefill token budget per iteration
    lookahead_tokens: int = 8  # KV headroom reserved for decode growth

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_num_batched_tokens < 1:
            raise ValueError("max_num_batched_tokens must be >= 1")


@dataclass
class AdmissionState:
    """Per-iteration view of resource availability."""

    running_count: int = 0
    free_blocks: int = 0
    block_size: int = 1
    remaining_token_budget: int = 0


@dataclass
class AdmissionCheck:
    ok: bool = False
    reason: str | None = None


class AdmissionPolicy:
    """FIFO admission with token-budget blocking and KV-capacity guard.

    A request is admitted only when *all* of the following hold:

    1. the running batch has space (``max_batch_size``);
    2. its prefill work (prompt tokens plus generated tokens for preempted
       sequences being re-prefilled) fits the iteration token budget;
    3. KV capacity covers the tokens plus decode headroom: fresh requests
       reserve ``lookahead_tokens`` of growth so an admitted sequence can
       take at least one decode step before admission pressure is
       reconsidered; preempted sequences (re-prefilled from prompt +
       generated tokens) reserve just one step, since their continuation
       capacity is re-evaluated per decode step by the engine.
    """

    def __init__(self, limits: SchedulerLimits) -> None:
        self.limits = limits

    def can_admit(self, request: GenerationRequest, state: AdmissionState) -> AdmissionCheck:
        check = AdmissionCheck()
        if state.running_count >= self.limits.max_batch_size:
            check.reason = "batch full"
            return check
        tokens = request.total_kv_tokens
        if tokens > state.remaining_token_budget:
            check.reason = "prefill token budget exhausted"
            return check
        headroom = 1 if request.preempted else self.limits.lookahead_tokens
        blocks_needed = (tokens + headroom + state.block_size - 1) // state.block_size
        if blocks_needed > state.free_blocks:
            check.reason = "insufficient KV blocks"
            return check
        check.ok = True
        return check
