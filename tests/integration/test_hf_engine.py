"""Real-model integration tests.

These tests exercise the full engine against an actual Hugging Face model
(Qwen/Qwen2.5-0.5B-Instruct by default). They are slow and require the
model to be present locally, so they are skipped unless the environment
variable FLUXLLM_INTEGRATION=1 is set and fluxllm[torch] is installed.
CI runs them in a dedicated optional job, never in the fast unit lane.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("FLUXLLM_INTEGRATION") != "1",
    reason="set FLUXLLM_INTEGRATION=1 (plus fluxllm[torch] extras) to run real-model tests",
)

MODEL = os.environ.get("FLUXLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def hf_settings():
    from fluxllm.config.settings import EngineSettings

    return EngineSettings(
        model_name=MODEL,
        dtype="float32",
        num_blocks=256,
        block_size=16,
        max_batch_size=4,
        max_num_batched_tokens=1024,
        max_new_tokens=32,
        max_prompt_tokens=256,
    )


async def test_hf_engine_generates_coherent_text():
    from fluxllm.engine.engine import FluxEngine
    from fluxllm.engine.lifecycle import RequestState
    from fluxllm.sampling.config import SamplingParams

    engine = FluxEngine.from_settings(hf_settings())
    engine.start()
    try:
        handle = engine.submit(
            "The capital of France is",
            SamplingParams(temperature=0.0, max_new_tokens=8),
        )
        tokens = []
        async for event in handle.stream():
            if event.type == "token" and event.token_id is not None:
                tokens.append(event.token_id)
        assert handle.request.state is RequestState.COMPLETED
        assert 1 <= len(tokens) <= 8
        text = engine._detokenize(handle.request.generated_token_ids)
        assert "Paris" in text, f"unexpected generation: {text!r}"
        assert engine.block_manager.used_blocks == 0
    finally:
        await engine.aclose()


async def test_hf_engine_concurrent_requests_release_blocks():
    import asyncio

    from fluxllm.engine.engine import FluxEngine
    from fluxllm.engine.lifecycle import RequestState
    from fluxllm.sampling.config import SamplingParams

    engine = FluxEngine.from_settings(hf_settings())
    engine.start()
    try:
        prompts = [
            "Briefly explain what a KV cache is.",
            "Briefly explain what tokenization is.",
            "Briefly explain what GPU memory is.",
        ]
        handles = [
            engine.submit(p, SamplingParams(temperature=0.0, max_new_tokens=12)) for p in prompts
        ]
        await asyncio.gather(*(_consume(h) for h in handles))
        for handle in handles:
            assert handle.request.state is RequestState.COMPLETED
        assert engine.block_manager.used_blocks == 0
        assert engine.stats()["kv_cache"]["utilization"] == 0.0
    finally:
        await engine.aclose()


async def _consume(handle) -> None:
    async for _event in handle.stream():
        pass


def _make_executor():
    from fluxllm.executor.hf_executor import HFModelExecutor

    return HFModelExecutor(model_name=MODEL, dtype="float32", device="cpu")


def test_hf_batched_decode_matches_sequential():
    from fluxllm.executor.base import SequenceState

    executor = _make_executor()
    try:
        prompts = ["The capital of France is", "The color of the sky is", "Two plus two is"]
        seq_ids = [executor.encode(p) for p in prompts]
        states = []
        seq_streams: list[list[int]] = []
        for i, pid in enumerate(seq_ids):
            out = executor.prefill(str(i), pid)
            states.append(out.state)
            stream: list[int] = [int(out.logits.argmax())]
            for _ in range(5):
                out = executor.decode(states[i], stream[-1])
                states[i] = out.state
                stream.append(int(out.logits.argmax()))
            seq_streams.append(stream[:5])
        # batched: same prompts, same greedy argmax picks
        executor.release("0")
        executor.release("1")
        executor.release("2")
        b_states = [SequenceState(request_id=str(i), num_tokens=len(seq_ids[i])) for i in range(3)]
        outs = executor.prefill_batch([(str(i), seq_ids[i]) for i in range(3)])
        assert [o.state.num_tokens for o in outs] == [len(pid) for pid in seq_ids]
        streams: list[list[int]] = []
        for i, out in enumerate(outs):
            b_states[i] = out.state
            streams.append([int(out.logits.argmax())])
        for _ in range(5):
            outs = executor.decode_batch([(b_states[i], streams[i][-1]) for i in range(3)])
            for i, out in enumerate(outs):
                b_states[i] = out.state
                streams[i].append(int(out.logits.argmax()))
        for i in range(3):
            assert streams[i][:5] == seq_streams[i], (
                f"batched decode diverged for prompt {prompts[i]!r}: "
                f"{streams[i][:5]} vs {seq_streams[i]}"
            )
    finally:
        executor.release("0")
        executor.release("1")
        executor.release("2")


def test_hf_batched_prefill_matches_sequential():

    executor = _make_executor()
    try:
        prompts = ["Rainbows form when", "Turtles are reptiles that", "Gravity bends light near"]
        ids = [executor.encode(p) for p in prompts]
        outs = executor.prefill_batch([(str(i), ids[i]) for i in range(3)])
        batched_argmax = [int(o.logits.argmax()) for o in outs]
        # KV state lengths must line up for every row (checked while the
        # batched rows are still registered)
        for i in range(3):
            entry = executor._caches.get(str(i))
            assert entry is not None and entry["num_tokens"] == len(ids[i])
            del executor._caches[str(i)]
        executor.release("0")
        executor.release("1")
        executor.release("2")
        sequential: list[int] = []
        for i, pid in enumerate(ids):
            out = executor.prefill(f"s{i}", pid)
            sequential.append(int(out.logits.argmax()))
            executor.release(f"s{i}")
        assert batched_argmax == sequential, "batched prefill first-token logits diverged"
    finally:
        executor.release("0")
        executor.release("1")
        executor.release("2")
