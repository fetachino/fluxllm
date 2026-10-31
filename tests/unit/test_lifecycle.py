"""Lifecycle transition table and request model tests."""

from __future__ import annotations

import pytest

from fluxllm.engine.lifecycle import (
    ALLOWED_TRANSITIONS,
    LifecycleError,
    RequestState,
    can_transition,
    validate_transition,
)
from fluxllm.engine.request import CompletionReason, GenerationRequest, RequestTimings
from fluxllm.sampling.config import SamplingParams


def test_terminal_states_absorb_everything():
    for terminal in (RequestState.COMPLETED, RequestState.CANCELLED, RequestState.FAILED):
        for other in RequestState:
            assert not can_transition(terminal, other)


def test_valid_happy_path():
    validate_transition(RequestState.QUEUED, RequestState.PREFILL)
    validate_transition(RequestState.PREFILL, RequestState.DECODING)
    validate_transition(RequestState.DECODING, RequestState.COMPLETED)


def test_preemption_loops_back_to_queue():
    assert can_transition(RequestState.DECODING, RequestState.QUEUED)
    assert can_transition(RequestState.PREFILL, RequestState.QUEUED)


def test_invalid_backward_transition_raises():
    with pytest.raises(LifecycleError):
        validate_transition(RequestState.DECODING, RequestState.PREFILL)


def test_transition_table_is_exhaustive():
    for state in RequestState:
        assert state in ALLOWED_TRANSITIONS


def make_request(prompt: str = "abc") -> GenerationRequest:
    return GenerationRequest(prompt=prompt, sampling=SamplingParams(max_new_tokens=4))


def run_to_decoding() -> GenerationRequest:
    request = make_request()
    request.transition(RequestState.PREFILL)
    request.transition(RequestState.DECODING)
    return request


def test_request_finish_sets_state_and_reason():
    request = run_to_decoding()
    request.finish(CompletionReason.LENGTH)
    assert request.state is RequestState.COMPLETED
    assert request.completion_reason is CompletionReason.LENGTH
    assert request.is_terminal


def test_finish_is_idempotent():
    request = run_to_decoding()
    request.finish(CompletionReason.LENGTH)
    request.finish(CompletionReason.CANCELLED)
    assert request.completion_reason is CompletionReason.LENGTH


def test_fail_sets_error_message():
    request = make_request()
    request.fail("boom")
    assert request.state is RequestState.FAILED
    assert request.error == "boom"


def test_timing_properties():
    timings = RequestTimings(submitted_at=0.0, admitted_at=1.0, first_token_at=2.0, finished_at=5.0)
    assert timings.queue_delay == 1.0
    assert timings.ttft == 2.0
    assert timings.prefill_latency == 1.0  # admission -> first token
    assert timings.decode_latency == 3.0
    assert timings.total_latency == 5.0
