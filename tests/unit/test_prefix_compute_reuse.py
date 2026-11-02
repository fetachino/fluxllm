"""Executor-level prefix compute-reuse tests (deterministic twin).

These exercise the reuse bookkeeping and metrics without PyTorch. Exact
logit equality after reuse is verified against the real model in
``tests/integration/test_hf_engine.py``.
"""

from __future__ import annotations

import numpy as np

from fluxllm.executor.deterministic import DeterministicExecutor
from fluxllm.metrics.metrics import EngineMetrics


def make_executor(**kwargs) -> DeterministicExecutor:
    return DeterministicExecutor(vocab_size=64, **kwargs)


def test_exact_match_reuses_the_whole_prompt():
    executor = make_executor()
    first = executor.prefill("a", [1, 2, 3, 4])
    second = executor.prefill("b", [1, 2, 3, 4])
    assert executor.prefix_reused_tokens == 4
    assert second.state.num_tokens == 4
    np.testing.assert_allclose(first.logits, second.logits)


def test_shorter_cached_prefix_is_reused():
    executor = make_executor()
    executor.prefill("a", [1, 2, 3])
    out = executor.prefill("b", [1, 2, 3, 4, 5])
    assert executor.prefix_reused_tokens == 3
    assert out.state.num_tokens == 5


def test_covering_prefix_reuses_all_but_the_last_token():
    executor = make_executor()
    executor.prefill("a", [1, 2, 3, 4, 5])
    out = executor.prefill("b", [1, 2, 3])
    assert executor.prefix_reused_tokens == 2
    assert out.state.num_tokens == 3


def test_no_reuse_for_unrelated_prompts():
    executor = make_executor()
    executor.prefill("a", [1, 2, 3])
    executor.prefill("b", [7, 8, 9])
    assert executor.prefix_reused_tokens == 0


def test_reuse_can_be_disabled():
    executor = make_executor(enable_prefix_compute_reuse=False)
    executor.prefill("a", [1, 2, 3])
    executor.prefill("b", [1, 2, 3])
    assert executor.prefix_reused_tokens == 0


def test_lru_evicts_old_snapshots():
    executor = make_executor(prefix_kv_entries=1)
    executor.prefill("a", [1, 2, 3])
    executor.prefill("b", [4, 5, 6])  # evicts the first snapshot
    executor.prefill("c", [1, 2, 3])
    assert executor.prefix_reused_tokens == 0


def test_metrics_record_reused_tokens():
    metrics = EngineMetrics()
    executor = make_executor(metrics=metrics)
    executor.prefill("a", [1, 2, 3, 4])
    executor.prefill("b", [1, 2, 3, 4, 5])
    assert metrics.prefix_compute_reuses._value.get() == 1
    assert metrics.prefix_compute_reused_tokens._value.get() == 4


def test_batch_prefill_reuses_per_row():
    metrics = EngineMetrics()
    executor = make_executor(metrics=metrics)
    executor.prefill("a", [1, 2, 3, 4])
    outs = executor.prefill_batch(
        [("b", [1, 2, 3, 4, 5]), ("c", [9, 9, 9])]
    )
    assert [out.state.num_tokens for out in outs] == [5, 3]
    assert metrics.prefix_compute_reuses._value.get() == 1
    assert metrics.prefix_compute_reused_tokens._value.get() == 4
