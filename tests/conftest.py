"""Shared pytest fixtures: a deterministic engine factory plus helpers."""

from __future__ import annotations

import pytest

from fluxllm.config.settings import EngineSettings
from fluxllm.engine.engine import FluxEngine
from fluxllm.executor.deterministic import DeterministicExecutor


@pytest.fixture
def engine_factory():
    """Build deterministic engines with custom pool/batch settings."""

    def _make(
        num_blocks: int = 16,
        block_size: int = 4,
        max_batch_size: int = 4,
        max_num_batched_tokens: int = 1000,
        max_queue_size: int = 16,
        max_prompt_tokens: int = 256,
        max_new_tokens: int = 64,
        vocab_size: int = 32,
        eos_token_id: int | None = None,
        enable_batched_execution: bool = True,
        enable_prefix_compute_reuse: bool = True,
        executor: DeterministicExecutor | None = None,
    ) -> FluxEngine:
        settings = EngineSettings(
            deterministic=True,
            vocab_size=vocab_size,
            num_blocks=num_blocks,
            block_size=block_size,
            max_batch_size=max_batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            max_queue_size=max_queue_size,
            max_prompt_tokens=max_prompt_tokens,
            max_new_tokens=max_new_tokens,
            enable_batched_execution=enable_batched_execution,
        )
        engine = FluxEngine.from_settings(
            settings,
            executor=executor
            or DeterministicExecutor(
                vocab_size=vocab_size,
                eos_token_id=eos_token_id,
                enable_prefix_compute_reuse=enable_prefix_compute_reuse,
            ),
        )
        return engine

    return _make


async def collect_tokens(handle) -> list[int]:
    """Consume a handle's stream and return generated token ids."""
    return [
        event.token_id
        for event in [e async for e in handle.stream()]
        if event is not None and getattr(event, "type", "") == "token"
    ]
