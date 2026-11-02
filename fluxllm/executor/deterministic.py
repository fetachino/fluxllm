"""Deterministic, dependency-light executor.

Produces reproducible fake logits from a fixed arithmetic pattern so
scheduler, cache, and engine tests run in milliseconds without PyTorch or a
model download. It is *not* a language model: it is used for engineering
simulation, and every interface that exposes it labels it as such.

Generation pattern: with KV length ``n``, the argmax token is
``(n * step + offset) % vocab``. The executor reports the executor-side
state after each step, exactly like the HF executor, so the engine treats
both identically.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fluxllm.executor.base import SequenceState, StepOutput


class DeterministicExecutor:
    """Fake executor with fully reproducible logits."""

    def __init__(
        self,
        vocab_size: int = 64,
        eos_token_id: int | None = None,
        step: int = 7,
        offset: int = 3,
        max_context_len: int = 8192,
    ) -> None:
        self._vocab_size = vocab_size
        self._eos_token_id = eos_token_id
        self._step = step
        self._offset = offset
        self._max_context_len = max_context_len
        self._states: dict[str, SequenceState] = {}
        # Uniform low logit for non-target entries so sampling policy paths
        # (temperature/top-k/top-p) see a non-degenerate distribution.
        self._noise_logit = -16.0

    # -- properties ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def max_context_len(self) -> int:
        return self._max_context_len

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        """Drop all sequence state (used between tests/simulations)."""
        self._states.clear()

    def release(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    # -- forward ------------------------------------------------------------
    def _pattern_token(self, kv_len: int) -> int:
        return (kv_len * self._step + self._offset) % self._vocab_size

    def _build_logits(self, state: SequenceState) -> StepOutput:
        target = self._pattern_token(state.num_tokens)
        logits = np.full(self._vocab_size, self._noise_logit, dtype=np.float32)
        logits[target] = 24.0
        return StepOutput(
            logits=logits,
            state=state,
            context_exceeded=state.num_tokens >= self._max_context_len,
        )

    def prefill(self, request_id: str, token_ids: Sequence[int]) -> StepOutput:
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        if request_id in self._states:
            raise RuntimeError(f"sequence {request_id} already prefilled")
        state = SequenceState(request_id=request_id, num_tokens=len(token_ids))
        self._states[request_id] = state
        return self._build_logits(state)

    def decode(self, state: SequenceState, last_token: int) -> StepOutput:
        cached = self._states.get(state.request_id)
        if cached is None or cached.num_tokens != state.num_tokens:
            # The engine is the single writer of SequenceState; a mismatch
            # means engine bookkeeping and executor state have diverged.
            raise RuntimeError(f"decode for unknown/stale sequence {state.request_id}")
        cached.num_tokens += 1
        state.num_tokens = cached.num_tokens
        return self._build_logits(cached)

    # -- batched execution (same logits pattern as the sequential path) ------
    def prefill_batch(self, prompts: Sequence[tuple[str, Sequence[int]]]) -> list[StepOutput]:
        """Prefill several prompts; validates every row before any state."""
        if not prompts:
            return []
        for request_id, tokens in prompts:
            if not tokens:
                raise ValueError("prefill requires at least one token")
            if request_id in self._states:
                raise RuntimeError(f"sequence {request_id} already prefilled")
        outs: list[StepOutput] = []
        for request_id, tokens in prompts:
            state = SequenceState(request_id=request_id, num_tokens=len(tokens))
            self._states[request_id] = state
            outs.append(self._build_logits(state))
        return outs

    def decode_batch(self, batch: Sequence[tuple[SequenceState, int]]) -> list[StepOutput]:
        """Advance several sequences one token; identical results to sequential."""
        if not batch:
            return []
        entries: list[tuple[SequenceState, SequenceState]] = []
        for state, _last_token in batch:
            cached = self._states.get(state.request_id)
            if cached is None or cached.num_tokens != state.num_tokens:
                raise RuntimeError(f"decode for unknown/stale sequence {state.request_id}")
            entries.append((cached, state))
        outs: list[StepOutput] = []
        for cached, state in entries:
            cached.num_tokens += 1
            state.num_tokens = cached.num_tokens
            outs.append(self._build_logits(cached))
        return outs
