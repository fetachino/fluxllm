"""Model execution abstraction.

FluxLLM separates *what runs the model* from *how requests are scheduled*.
A ModelExecutor advances one sequence step at a time and returns the next
token's logits. Concrete implementations:

- :class:`fluxllm.executor.deterministic.DeterministicExecutor` — numpy-only,
  dependency-free, reproducible; used by unit tests and scheduler simulation.
- :class:`fluxllm.executor.hf_executor.HFModelExecutor` — runs Hugging Face
  causal language models on CPU/CUDA with a real single-sequence KV cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class SequenceState:
    """Per-sequence execution state the executor needs to continue decoding.

    ``kv`` is executor-opaque: the HF executor stores torch tensors (its own
    KV cache), the deterministic executor stores bookkeeping only. The
    scheduler and cache manager never inspect this field.
    """

    request_id: str
    num_tokens: int  # tokens whose KV entries the executor holds
    kv: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredToken:
    """A sampled token with its selection probability (for logging/metrics)."""

    token_id: int
    probability: float


@dataclass
class StepOutput:
    """Result of advancing one sequence one position."""

    # Logits for the next token position, shape (vocab,), unnormalized.
    logits: np.ndarray
    # Updated executor-side state for the sequence (post-step KV length).
    state: SequenceState | None = None
    # True when the executor already knows the sequence cannot continue
    # (e.g. context limit reached). The engine will fail/complete the request.
    context_exceeded: bool = False


@runtime_checkable
class ModelExecutor(Protocol):
    """A single-sequence forward executor.

    The executor creates/extends per-sequence KV state and executes exactly
    one forward per call. Batching across sequences is a scheduler-level
    concept in FluxLLM; see docs (decode forwards are per-sequence, unlike
    production engines that batch the attention kernel).
    """

    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def max_context_len(self) -> int: ...

    def prefill(self, request_id: str, token_ids: Sequence[int]) -> StepOutput:
        """Process a full prompt, returning logits for the next-to-sample position."""
        ...

    def decode(self, state: SequenceState, last_token: int) -> StepOutput:
        """Advance a sequence by ``last_token`` (previously sampled), return next logits."""
        ...

    def release(self, request_id: str) -> None:
        """Free any executor-side resources held for the request."""
        ...
