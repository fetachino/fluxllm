"""Sampling parameter model."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SamplingParams:
    """Controls how the sampler selects the next token.

    ``temperature == 0`` means greedy decoding. ``top_k``/``top_p`` operate
    on temperature-scaled probabilities. ``seed`` makes a request's sampling
    reproducible; use ``None`` for nondeterministic sampling.
    """

    max_new_tokens: int = 64
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0
