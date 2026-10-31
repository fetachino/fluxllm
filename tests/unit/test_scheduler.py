"""Scheduler tests: admission rules, FIFO blocking, preemption, bookkeeping."""

from __future__ import annotations

import pytest

from fluxllm.cache.block_manager import BlockManager
from fluxllm.engine.lifecycle import RequestState
from fluxllm.engine.request import CompletionReason, GenerationRequest
from fluxllm.sampling.config import SamplingParams
from fluxllm.scheduler.policy import SchedulerLimits
from fluxllm.scheduler.scheduler import Scheduler

VOCAB = 32


def tokenize_words(text: str) -> list[int]:
    from fluxllm.engine.tokens import simple_tokenize

    return simple_tokenize(text, VOCAB)


def make_scheduler(num_blocks=16, block_size=4, max_batch_size=4, max_num_batched_tokens=1000):
    blocks = BlockManager(num_blocks=num_blocks, block_size=block_size)
    limits = SchedulerLimits(
        max_batch_size=max_batch_size,
        max_num_batched_tokens=max_num_batched_tokens,
        lookahead_tokens=2,
    )
    return Scheduler(blocks, limits)


def submit(scheduler: Scheduler, prompt: str, max_new: int = 4) -> GenerationRequest:
    request = GenerationRequest(prompt=prompt, sampling=SamplingParams(max_new_tokens=max_new))
    request.prompt_token_ids = tokenize_words(prompt)
    assert scheduler.submit(request)
    return request


def test_submitted_requests_are_queued():
    scheduler = make_scheduler()
    request = submit(scheduler, "alpha beta")
    assert scheduler.queued_count == 1
    assert request.state is RequestState.QUEUED


def test_plan_admits_fifo_requests_and_allocates_blocks():
    scheduler = make_scheduler()
    r1 = submit(scheduler, "alpha beta gamma")
    r2 = submit(scheduler, "delta epsilon")
    plan = scheduler.plan()
    assert [r.request_id for r in plan.prefills] == [r1.request_id, r2.request_id]
    assert r1.state is RequestState.PREFILL
    assert r2.state is RequestState.PREFILL
    assert scheduler.running_count == 2
    assert scheduler.blocks.has_sequence(r1.request_id)
    scheduler.blocks.check_invariants()


def test_batch_size_limit_blocks_admission():
    scheduler = make_scheduler(max_batch_size=1)
    r1 = submit(scheduler, "alpha beta")
    r2 = submit(scheduler, "gamma delta")
    plan = scheduler.plan()
    assert len(plan.prefills) == 1
    assert plan.prefills[0].request_id == r1.request_id
    assert scheduler.queued_count == 1
    assert r2.state is RequestState.QUEUED


def test_token_budget_blocks_second_admission():
    scheduler = make_scheduler(max_num_batched_tokens=5)
    submit(scheduler, "alpha beta gamma")  # 3 tokens
    submit(scheduler, "delta epsilon zeta eta")  # 4 tokens: 3+4 > budget 5
    plan = scheduler.plan()
    assert len(plan.prefills) == 1
    assert scheduler.queued_count == 1


def test_insufficient_blocks_block_admission_fifo():
    scheduler = make_scheduler(num_blocks=3, block_size=4)
    r1 = submit(scheduler, "alpha beta")  # 2 tokens -> 1 block
    # 9 tokens: needs 3 blocks at admission (with lookahead) but only 2 free
    submit(scheduler, "w1 w2 w3 w4 w5 w6 w7 w8 w9")
    submit(scheduler, "zeta eta")  # small enough to fit, but FIFO blocks behind r2
    plan = scheduler.plan()
    assert len(plan.prefills) == 1
    assert plan.prefills[0].request_id == r1.request_id
    assert scheduler.queued_count == 2


def test_queue_full_rejects_submission():
    scheduler = make_scheduler(max_batch_size=8)
    scheduler.max_queue_size = 1
    submit(scheduler, "alpha beta")
    request2 = GenerationRequest(prompt="x y", sampling=SamplingParams())
    request2.prompt_token_ids = [1]
    assert scheduler.submit(request2) is False


def test_submit_requires_tokenization():
    scheduler = make_scheduler()
    request = GenerationRequest(prompt="x", sampling=SamplingParams())
    with pytest.raises(ValueError):
        scheduler.submit(request)


def test_preempt_requeues_and_frees_blocks():
    scheduler = make_scheduler()
    request = submit(scheduler, "alpha beta gamma delta")
    plan = scheduler.plan()
    assert len(plan.prefills) == 1
    scheduler.preempt(request)
    assert request.state is RequestState.QUEUED
    assert request.preempted is True
    assert scheduler.queued_count == 1
    assert scheduler.running_count == 0
    assert scheduler.blocks.free_blocks == scheduler.blocks.num_blocks
    assert request.timings.admitted_at is not None  # first admission recorded


def test_preempted_request_readmitted_with_single_step_headroom():
    scheduler = make_scheduler(num_blocks=4, block_size=4)
    request = submit(scheduler, "alpha beta gamma delta epsilon", max_new=8)
    plan = scheduler.plan()
    assert len(plan.prefills) == 1
    scheduler.blocks.extend(request.request_id)  # simulate several decodes
    scheduler.blocks.extend(request.request_id)
    scheduler.blocks.extend(request.request_id)
    scheduler.blocks.extend(request.request_id)
    scheduler.preempt(request)
    # 5 prompt + 4 generated = 9 KV tokens = 3 blocks of 4; pool has 4 → fits
    plan2 = scheduler.plan()
    assert len(plan2.prefills) == 1
    assert request.state is RequestState.PREFILL


def test_complete_releases_everything():
    scheduler = make_scheduler()
    request = submit(scheduler, "alpha beta")
    scheduler.plan()
    assert request.state is RequestState.PREFILL
    scheduler.complete(request, CompletionReason.LENGTH)
    assert request.is_terminal
    assert scheduler.running_count == 0
    assert scheduler.blocks.free_blocks == scheduler.blocks.num_blocks


def test_cancel_queued_request_removes_from_queue():
    scheduler = make_scheduler(max_batch_size=1)
    submit(scheduler, "alpha beta")
    r2 = submit(scheduler, "gamma delta")
    scheduler.plan()
    assert scheduler.queued_count == 1
    scheduler.cancel(r2)
    assert scheduler.queued_count == 0
    assert r2.state is RequestState.CANCELLED


def test_advance_reserves_kv_slots():
    scheduler = make_scheduler()
    request = submit(scheduler, "alpha beta")
    scheduler.plan()
    current = scheduler.blocks.sequence_kv_tokens(request.request_id)
    assert scheduler.advance(request)
    assert scheduler.blocks.sequence_kv_tokens(request.request_id) == current + 1


def test_iteration_counter_increments():
    scheduler = make_scheduler()
    submit(scheduler, "alpha beta")
    scheduler.plan()
    scheduler.plan()
    assert scheduler.iteration_count == 2
