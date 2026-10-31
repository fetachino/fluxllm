"""Sampler tests: greedy, temperature, top-k, top-p, and determinism."""

from __future__ import annotations

import numpy as np
import pytest

from fluxllm.sampling.config import SamplingParams
from fluxllm.sampling.sampler import Sampler, _top_p_mask


def logits_map(**overrides) -> np.ndarray:
    """A distribution peaked at index 3 with a long tail."""
    base = np.array([-5.0, -3.0, -4.0, 2.0, -1.0, -6.0, -2.5, -5.5], dtype=np.float32)
    for index, value in overrides.items():
        base[int(index[1:])] = value
    return base


def test_greedy_picks_argmax():
    s = Sampler()
    assert s.sample(logits_map(), SamplingParams(temperature=0.0), np.random.default_rng(0)) == 3


def test_sampling_respects_seed():
    s = Sampler()
    params = SamplingParams(temperature=1.2, seed=1234)
    first = [s.sample(logits_map(), params, np.random.default_rng(1234)) for _ in range(5)]
    second = [s.sample(logits_map(), params, np.random.default_rng(1234)) for _ in range(5)]
    assert first == second  # same seed => identical sequence


def test_different_seeds_diverge_statistically():
    s = Sampler()
    params = SamplingParams(temperature=1.5)
    outcomes = {s.sample(logits_map(), params, np.random.default_rng(seed)) for seed in range(200)}
    assert len(outcomes) > 1


def test_top_k_restricts_candidates():
    s = Sampler()
    params = SamplingParams(temperature=1.0, top_k=1)  # equivalent to greedy
    got = {s.sample(logits_map(), params, np.random.default_rng(seed)) for seed in range(50)}
    assert got == {3}


def test_top_p_concentrates_mass_on_tail_below_p():
    s = Sampler()
    params = SamplingParams(temperature=1.0, top_p=0.3)
    # With a 2.0 peak and neighbor at -1.0, top_p=0.3 keeps only the peak.
    got = {s.sample(logits_map(), params, np.random.default_rng(seed)) for seed in range(50)}
    assert got == {3}


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.5)
    with pytest.raises(ValueError):
        SamplingParams(top_k=0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(max_new_tokens=0)


def test_top_p_mask_keeps_first_crossing_token():
    probs = np.array([0.5, 0.3, 0.2])
    masked = _top_p_mask(probs, 0.6)
    # 0.5 alone is below 0.6, so 0.5+0.3 set is kept (first crossing included)
    assert masked[0] > 0
    assert masked[1] > 0
    assert masked[2] == 0


def test_stop_token_membership():
    from fluxllm.sampling.sampler import is_stop_token

    assert is_stop_token(7, (7, 9))
    assert not is_stop_token(8, (7, 9))
