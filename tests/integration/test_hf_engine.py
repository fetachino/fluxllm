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
