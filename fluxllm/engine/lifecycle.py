"""Request lifecycle model for FluxLLM.

Every generation request moves through an explicit, validated state machine:

    QUEUED -> PREFILL -> DECODING -> COMPLETED
                            |
                           CANCELLED / FAILED

QUEUED and PREFILL may loop back through admission (a preempted sequence is
re-queued). Terminal states are COMPLETED, CANCELLED, and FAILED. All
transitions are validated so illegal moves raise immediately instead of
silently corrupting engine bookkeeping.
"""

from __future__ import annotations

import enum


class RequestState(enum.StrEnum):
    """Lifecycle states for a generation request."""

    QUEUED = "queued"
    PREFILL = "prefill"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class LifecycleError(RuntimeError):
    """Raised when an invalid state transition is attempted."""


#: Explicit transition table. Validation lives here, not scattered across the
#: scheduler, so the lifecycle is testable in isolation.
ALLOWED_TRANSITIONS: dict[RequestState, frozenset[RequestState]] = {
    RequestState.QUEUED: frozenset(
        {RequestState.PREFILL, RequestState.CANCELLED, RequestState.FAILED}
    ),
    RequestState.PREFILL: frozenset(
        {
            RequestState.DECODING,
            RequestState.COMPLETED,  # immediate stop after prefill (e.g. instant EOS)
            RequestState.QUEUED,  # preempted back to queue (KV recomputed on readmission)
            RequestState.CANCELLED,
            RequestState.FAILED,
        }
    ),
    RequestState.DECODING: frozenset(
        {
            RequestState.COMPLETED,
            RequestState.QUEUED,  # preempted back to queue (KV recomputed on readmission)
            RequestState.CANCELLED,
            RequestState.FAILED,
        }
    ),
    RequestState.COMPLETED: frozenset(),
    RequestState.CANCELLED: frozenset(),
    RequestState.FAILED: frozenset(),
}


def can_transition(current: RequestState, target: RequestState) -> bool:
    """Return True when moving from ``current`` to ``target`` is legal."""
    return target in ALLOWED_TRANSITIONS[current]


def validate_transition(current: RequestState, target: RequestState) -> None:
    """Raise :class:`LifecycleError` unless the transition is legal."""
    if not can_transition(current, target):
        raise LifecycleError(f"illegal request state transition: {current.value} -> {target.value}")
