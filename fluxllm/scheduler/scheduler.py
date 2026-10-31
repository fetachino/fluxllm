"""Continuous batching scheduler.

Owns the queue of pending requests and the set of running sequences. Each
engine iteration asks the scheduler for a *plan*: which sequences should be
prefilled (new admissions) and which should decode (continuing work). The
engine executes the plan with the model executor, then reports results back
through the bookkeeping methods below.

Scheduler/engine contract:

- ``plan()`` pre-allocates KV blocks for admitted requests;
- ``record_prefill`` registers the executor state for a prefilled sequence;
- ``advance`` consumes one KV slot for the token about to be fed to the
  executor, returning False when the pool is exhausted (the engine must
  then roll the sampled token back and preempt the sequence);
- ``preempt`` releases resources and re-queues the request (its KV state is
  recreated by re-prefilling prompt + already-generated tokens on its next
  admission — documented recompute behavior);
- ``complete``/``cancel``/``fail`` release resources for terminal requests.

Key property: requests arriving while others are mid-decode are admitted in
later iterations rather than waiting for the current batch to drain
(continuous batching). Fairness is strict FIFO (see policy.py).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from fluxllm.cache.block_manager import BlockManager, CacheFullError
from fluxllm.engine.lifecycle import RequestState
from fluxllm.engine.request import CompletionReason, GenerationRequest
from fluxllm.scheduler.policy import AdmissionPolicy, AdmissionState, SchedulerLimits

logger = logging.getLogger(__name__)


@dataclass
class IterationPlan:
    """Work selected for one engine iteration."""

    prefills: list[GenerationRequest] = field(default_factory=list)
    decodes: list[GenerationRequest] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.prefills and not self.decodes


@dataclass
class RunningSequence:
    """Bookkeeping for one admitted request."""

    request: GenerationRequest
    # Executor-side state returned by prefill (opaque to the scheduler).
    state: object | None = None


class Scheduler:
    """Admission + per-step planning around a running set of sequences."""

    def __init__(
        self,
        block_manager: BlockManager,
        limits: SchedulerLimits,
        max_queue_size: int = 128,
        on_queue_change: Callable[[int], None] | None = None,
    ) -> None:
        self.blocks = block_manager
        self.limits = limits
        self.policy = AdmissionPolicy(limits)
        self.max_queue_size = max_queue_size
        self.queue: list[GenerationRequest] = []
        self.running: dict[str, RunningSequence] = {}
        self.iteration_count = 0
        self.total_admissions = 0
        self.total_preemptions = 0
        self._on_queue_change = on_queue_change

    # -- request intake -------------------------------------------------------
    def submit(self, request: GenerationRequest) -> bool:
        """Queue a tokenized request. Returns False when the queue is full."""
        if not request.prompt_token_ids:
            raise ValueError("request must be tokenized before submission")
        if len(self.queue) >= self.max_queue_size:
            return False
        self.queue.append(request)
        self._notify_queue()
        return True

    # -- planning -------------------------------------------------------------
    def plan(self) -> IterationPlan:
        """Select and commit this iteration's prefill admissions + decode set."""
        self.iteration_count += 1
        plan = IterationPlan(decodes=[seq.request for seq in self.running.values()])

        budget = self.limits.max_num_batched_tokens
        remaining: list[GenerationRequest] = []
        blocked = False
        for request in self.queue:
            if request.state is not RequestState.QUEUED:
                continue  # drop terminalized stragglers
            if not blocked:
                state = AdmissionState(
                    running_count=self.running_count,
                    free_blocks=self.blocks.free_blocks,
                    block_size=self.blocks.block_size,
                    remaining_token_budget=budget,
                )
                check = self.policy.can_admit(request, state)
                if check.ok and self.blocks.can_allocate(request.total_kv_tokens):
                    self._admit(request, plan)
                    budget -= request.total_kv_tokens
                    continue
                logger.debug(
                    "admission blocked for %s: %s",
                    request.request_id,
                    check.reason or "insufficient KV blocks",
                )
                blocked = True  # strict FIFO: everyone behind waits too
            remaining.append(request)

        self.queue = remaining
        self._notify_queue()
        return plan

    def _admit(self, request: GenerationRequest, plan: IterationPlan) -> None:
        kv_tokens = request.total_kv_tokens
        self.blocks.allocate(request.request_id, kv_tokens, tuple(request.prompt_token_ids))
        request.transition(RequestState.PREFILL)
        if request.timings.admitted_at is None:
            request.time_to_admit()
        self.running[request.request_id] = RunningSequence(request=request)
        self.total_admissions += 1
        plan.prefills.append(request)

    # -- step bookkeeping ------------------------------------------------------
    def record_prefill(self, request: GenerationRequest, state: object) -> None:
        seq = self.running.get(request.request_id)
        if seq is None:
            raise RuntimeError(f"prefill result for unknown running request {request.request_id}")
        seq.state = state

    def sequence_state(self, request_id: str) -> object | None:
        seq = self.running.get(request_id)
        return seq.state if seq else None

    def advance(self, request: GenerationRequest) -> bool:
        """Reserve one KV slot for the token about to be fed to the executor.

        Returns False when the pool is exhausted (caller rolls back the
        sampled token and preempts the sequence).
        """
        try:
            self.blocks.extend(request.request_id)
            return True
        except CacheFullError:
            return False

    # -- preemption / completion ------------------------------------------------
    def preempt(self, request: GenerationRequest) -> None:
        """Return a running request to the queue, freeing its blocks.

        The request keeps its generated tokens; on re-admission the engine
        re-prefills prompt + generated tokens (KV recompute, vLLM-style).
        """
        self.running.pop(request.request_id, None)
        self.blocks.release(request.request_id)
        request.preempted = True
        request.transition(RequestState.QUEUED)
        self.queue.append(request)
        self.total_preemptions += 1
        self._notify_queue()

    def complete(self, request: GenerationRequest, reason: CompletionReason) -> None:
        if not request.is_terminal:
            request.finish(reason)
        self._release(request)

    def fail(self, request: GenerationRequest, message: str) -> None:
        request.fail(message)
        self._release(request)

    def cancel(self, request: GenerationRequest) -> None:
        if not request.is_terminal:
            request.finish(CompletionReason.CANCELLED)
        self._release(request)

    def _release(self, request: GenerationRequest) -> None:
        """Drop all resource holds (blocks, running set, queue membership)."""
        self.running.pop(request.request_id, None)
        self.blocks.release(request.request_id)
        for index, queued in enumerate(self.queue):
            if queued.request_id == request.request_id:
                del self.queue[index]
                break
        self._notify_queue()

    # -- stats ----------------------------------------------------------------
    @property
    def queued_count(self) -> int:
        return len(self.queue)

    @property
    def running_count(self) -> int:
        return len(self.running)

    def batch_size(self) -> int:
        return len(self.running)

    def _notify_queue(self) -> None:
        if self._on_queue_change is not None:
            self._on_queue_change(self.queued_count)
