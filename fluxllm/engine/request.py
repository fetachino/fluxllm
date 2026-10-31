"""Generation request model."""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fluxllm.engine.lifecycle import RequestState, validate_transition

if TYPE_CHECKING:
    from fluxllm.sampling.config import SamplingParams


class CompletionReason(enum.StrEnum):
    """Why a request stopped generating."""

    EOS = "eos_token"
    LENGTH = "length"
    STOP_TOKEN = "stop_token"
    STOP_STRING = "stop_string"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class RequestTimings:
    """Wall-clock timing checkpoints for a request.

    Timestamps are stored as ``time.perf_counter()`` values from the engine
    process. Derived intervals (queue delay, TTFT, decode span) are computed
    lazily so a single source of truth is kept for each measurement.
    """

    submitted_at: float = field(default_factory=time.perf_counter)
    admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None

    @property
    def queue_delay(self) -> float | None:
        """Time spent queued before the scheduler first admitted the request."""
        if self.admitted_at is None:
            return None
        return self.admitted_at - self.submitted_at

    @property
    def ttft(self) -> float | None:
        """Time to first token (includes queue delay and prefill)."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.submitted_at

    @property
    def prefill_latency(self) -> float | None:
        """Time from admission to first sampled token."""
        if self.admitted_at is None or self.first_token_at is None:
            return None
        return self.first_token_at - self.admitted_at

    @property
    def decode_latency(self) -> float | None:
        """Time from first token to completion."""
        if self.first_token_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.first_token_at

    @property
    def total_latency(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.submitted_at


@dataclass
class GenerationRequest:
    """A single unit of generation work moving through the engine.

    The request owns its lifecycle state, tokenization, sampling parameters,
    generated output, and timing metrics. The scheduler, cache manager, and
    executor all operate on this object without owning it.
    """

    prompt: str
    sampling: SamplingParams
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    state: RequestState = RequestState.QUEUED
    completion_reason: CompletionReason | None = None
    timings: RequestTimings = field(default_factory=RequestTimings)
    error: str | None = None
    # Monotonic per-request step counter of decode iterations completed
    # (used for determinism checks and per-token latency accounting).
    decode_steps: int = 0
    # Set by the scheduler when a preempted sequence is re-queued; any KV
    # blocks held before preemption have already been released.
    preempted: bool = False
    # Cancellation flag set by the client handle; consumed by the engine at
    # the next step boundary.
    cancel_requested: bool = False

    # -- lifecycle ---------------------------------------------------------
    def transition(self, target: RequestState) -> None:
        validate_transition(self.state, target)
        self.state = target

    @property
    def is_terminal(self) -> bool:
        return self.state in (
            RequestState.COMPLETED,
            RequestState.CANCELLED,
            RequestState.FAILED,
        )

    @property
    def is_running(self) -> bool:
        return self.state in (RequestState.PREFILL, RequestState.DECODING)

    # -- token accounting ---------------------------------------------------
    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    def append_token(self, token_id: int) -> None:
        self.generated_token_ids.append(token_id)

    @property
    def total_kv_tokens(self) -> int:
        """Tokens whose KV entries must be resident for this sequence."""
        return self.num_prompt_tokens + self.num_generated_tokens

    def time_to_admit(self) -> None:
        self.timings.admitted_at = time.perf_counter()

    def mark_first_token(self) -> None:
        if self.timings.first_token_at is None:
            self.timings.first_token_at = time.perf_counter()

    def finish(self, reason: CompletionReason) -> None:
        if self.is_terminal:
            return
        self.completion_reason = reason
        self.timings.finished_at = time.perf_counter()
        self.transition(
            RequestState.CANCELLED
            if reason is CompletionReason.CANCELLED
            else (
                RequestState.FAILED if reason is CompletionReason.ERROR else RequestState.COMPLETED
            )
        )

    def fail(self, message: str) -> None:
        if self.is_terminal:
            return
        self.error = message
        self.timings.finished_at = time.perf_counter()
        self.completion_reason = CompletionReason.ERROR
        self.transition(RequestState.FAILED)
