"""Engine integration tests on the deterministic executor (no torch needed).

Covers the continuous-batching behaviors that matter: shared iterations for
differently-timed arrivals, cancellation, prompt-limit failure, queue-full
rejection, and clean resource release.
"""

from __future__ import annotations

import asyncio

import pytest

from fluxllm.engine.engine import EngineFullError
from fluxllm.engine.lifecycle import RequestState
from fluxllm.sampling.config import SamplingParams


async def tokens_of(handle) -> list[int]:
    return [
        event.token_id
        async for event in handle.stream()
        if event.type == "token" and event.token_id is not None
    ]


async def test_single_request_completes(engine_factory):
    engine = engine_factory(num_blocks=8, block_size=4, max_new_tokens=12)
    engine.start()
    handle = engine.submit("hello world of batched tokens")
    events = [event async for event in handle.stream()]
    request = handle.request
    assert request.state is RequestState.COMPLETED
    assert request.completion_reason.value in ("length", "eos_token")
    assert len(request.generated_token_ids) == 12
    assert events[-1].type == "status"
    assert events[-1].state == "completed"
    assert all(event.type == "token" for event in events[:-1])
    await engine.aclose()


async def test_later_arrivals_share_engine_iterations(engine_factory):
    """Requests entering at different times must share execution.

    The late request is admitted, prefilled, and finishes while the long
    early request is still decoding — the defining behavior of continuous
    batching (no wait-for-batch-drain).
    """
    engine = engine_factory(num_blocks=16, block_size=4, max_batch_size=4)
    engine.start()

    early = engine.submit("long running request", SamplingParams(max_new_tokens=30))
    await asyncio.sleep(0.003)  # let the early request start decoding
    late = engine.submit("late arrival", SamplingParams(max_new_tokens=8))

    await asyncio.gather(tokens_of(early), tokens_of(late))

    assert late.request.timings.admitted_at is not None
    assert late.request.timings.finished_at < early.request.timings.finished_at
    # The late request was admitted while the early one was still running.
    assert late.request.timings.admitted_at < early.request.timings.finished_at
    assert engine.scheduler.total_preemptions == 0
    assert engine.block_manager.used_blocks == 0
    await engine.aclose()


async def test_cancellation_terminates_stream(engine_factory):
    engine = engine_factory(num_blocks=8, block_size=4, max_new_tokens=50)
    engine.start()
    handle = engine.submit("cancel me please", SamplingParams(max_new_tokens=50))
    seen = 0
    final_reason = None
    async for event in handle.stream():
        if event.type == "token":
            seen += 1
            if seen >= 3:
                handle.cancel()
        elif event.type == "status":
            final_reason = event.reason
    assert final_reason == "cancelled"
    assert handle.request.state is RequestState.CANCELLED
    assert seen < 50
    assert engine.block_manager.free_blocks == engine.block_manager.num_blocks
    await engine.aclose()


async def test_prompt_over_limit_fails_fast(engine_factory):
    engine = engine_factory(num_blocks=8, block_size=4, max_prompt_tokens=5)
    engine.start()
    handle = engine.submit("one two three four five six seven")
    events = [event async for event in handle.stream()]
    assert handle.request.state is RequestState.FAILED
    assert "token limit" in handle.request.error
    assert events[-1].reason == "error"
    await engine.aclose()


async def test_eos_terminates_generation(engine_factory):
    engine = engine_factory(num_blocks=8, block_size=4, vocab_size=32, eos_token_id=9)
    engine.start()
    for _ in range(3):
        handle = engine.submit("x y", SamplingParams(max_new_tokens=50))
        await tokens_of(handle)
        assert handle.request.state is RequestState.COMPLETED
        assert handle.request.completion_reason.value == "eos_token"
        assert handle.request.generated_token_ids[-1] == 9
    await engine.aclose()


async def test_queue_full_rejected(engine_factory):
    engine = engine_factory(max_queue_size=1, max_new_tokens=4)
    engine.start()
    handle = engine.submit("first request")
    first_task = asyncio.create_task(tokens_of(handle))
    with pytest.raises(EngineFullError):
        engine.submit("second request")
    await first_task
    await engine.aclose()


async def test_max_new_tokens_clamped_by_settings(engine_factory):
    engine = engine_factory(num_blocks=8, block_size=4, max_new_tokens=6)
    engine.start()
    handle = engine.submit("x y z", SamplingParams(max_new_tokens=100))
    await tokens_of(handle)
    assert len(handle.request.generated_token_ids) == 6
    await engine.aclose()


async def test_block_release_on_completion(engine_factory):
    engine = engine_factory(num_blocks=4, block_size=4, max_new_tokens=10)
    engine.start()
    for _ in range(3):
        handle = engine.submit("some prompt words here")
        await tokens_of(handle)
        assert engine.block_manager.used_blocks == 0
    await engine.aclose()


async def test_engine_stats_shape(engine_factory):
    engine = engine_factory()
    engine.start()
    handle = engine.submit("stats check")
    await tokens_of(handle)
    stats = engine.stats()
    assert stats["requests_queued"] == 0
    assert stats["requests_running"] == 0
    assert stats["total_admissions"] == 1
    assert stats["kv_cache"]["num_blocks"] == 16
    await engine.aclose()


async def test_preempted_request_recovers_without_duplication_under_pressure(
    engine_factory,
):
    """Under a tight pool, a running sequence must still finish exactly once."""
    engine = engine_factory(num_blocks=6, block_size=4, max_new_tokens=50)
    engine.start()
    # Two long generations against a small pool force preemption or blocking.
    handles = [
        engine.submit("alpha beta gamma delta epsilon", SamplingParams(max_new_tokens=6))
        for _ in range(2)
    ]
    outs = await asyncio.gather(*(tokens_of(h) for h in handles))
    for handle, tokens in zip(handles, outs, strict=False):
        assert handle.request.state is RequestState.COMPLETED
        assert len(tokens) == 6
        assert len(handle.request.generated_token_ids) == len(tokens)
    assert engine.block_manager.used_blocks == 0
    await engine.aclose()


async def test_repeat_prompts_reuse_prefix_blocks(engine_factory):
    """Identical prompts served back-to-back share full prefix blocks."""
    engine = engine_factory(num_blocks=16, block_size=4, max_new_tokens=2)
    engine.start()
    for _ in range(2):
        handle = engine.submit("alpha beta gamma delta", SamplingParams(max_new_tokens=2))
        await tokens_of(handle)
    cache = engine.block_manager.prefix_cache
    assert cache is not None
    assert cache.hits >= 1
    assert cache.reused_tokens > 0
    await engine.aclose()


async def test_back_to_back_engines_stream_normally(engine_factory):
    """Regression: cross-engine metrics state must never strand admissions.

    Repeated engine builds in one process (tests, API reloads) share the
    metrics singleton while each block manager restarts its counters at
    zero. With per-event prefix recording, every engine must stream.
    """
    for _ in range(2):
        engine = engine_factory(num_blocks=8, block_size=4, max_new_tokens=4)
        engine.start()
        handle = engine.submit("alpha beta gamma delta epsilon", SamplingParams(max_new_tokens=4))
        events = await asyncio.wait_for(tokens_of(handle), timeout=10)
        assert len(events) == 4
        assert handle.request.state is RequestState.COMPLETED
        await engine.aclose()
