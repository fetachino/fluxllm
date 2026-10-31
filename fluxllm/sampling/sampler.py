"""Token sampling, separated from model execution.

Pure numpy so it works identically for every executor: the HF executor
converts torch logits to numpy, the deterministic executor produces numpy
directly. ``sample`` implements greedy decoding, temperature scaling,
top-k, and top-p (nucleus) filtering with per-request seeding.
"""

from __future__ import annotations

import numpy as np

from fluxllm.sampling.config import SamplingParams


class Sampler:
    """Stateless sampler; per-request determinism comes from ``rng``."""

    def sample(self, logits: np.ndarray, params: SamplingParams, rng: np.random.Generator) -> int:
        if params.greedy:
            return int(np.argmax(logits))
        scaled = logits.astype(np.float64) / params.temperature
        if params.top_k < scaled.size:
            kth = np.partition(scaled, -params.top_k)[-params.top_k]
            scaled = np.where(scaled < kth, -np.inf, scaled)
        probs = _softmax(scaled)
        if params.top_p < 1.0:
            probs = _top_p_mask(probs, params.top_p)
        return int(rng.choice(probs.size, p=probs))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def _top_p_mask(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Keep the smallest set of tokens whose cumulative probability >= top_p."""
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    keep = cumulative - probs[order] < top_p  # first token crossing top_p stays
    if not keep.any():
        keep[0] = True
    masked = np.zeros_like(probs)
    masked[order[keep]] = probs[order[keep]] / probs[order[keep]].sum()
    return masked


def is_stop_token(token_id: int, stop_token_ids: tuple[int, ...]) -> bool:
    return token_id in stop_token_ids
