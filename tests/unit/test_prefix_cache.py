"""Prefix-cache accounting tests (logical reuse at the block level)."""

from __future__ import annotations

import pytest

from fluxllm.cache.block_manager import BlockManager, hash_tokens
from fluxllm.metrics.metrics import EngineMetrics


def make_manager(enable=True, num_blocks=8, block_size=4):
    return BlockManager(num_blocks=num_blocks, block_size=block_size, enable_prefix_cache=enable)


def test_identical_prompts_share_blocks():
    manager = make_manager(num_blocks=8, block_size=4)
    tokens = tuple(range(8))  # two full blocks
    manager.allocate("r1", kv_tokens=8, token_ids=tokens)
    manager.allocate("r2", kv_tokens=8, token_ids=tokens)
    # Both sequences hold 2 blocks, but only 2 distinct blocks are rented.
    assert manager.used_blocks == 2
    cache = manager.prefix_cache
    assert cache.hits == 2
    assert cache.reused_tokens == 8
    assert cache.hit_rate == pytest.approx(0.5)  # 2 hits / 4 lookups
    manager.check_invariants()


def test_partial_prefix_shares_only_matching_blocks():
    manager = make_manager(num_blocks=8, block_size=4)
    shared = tuple(range(4))
    manager.allocate("r1", kv_tokens=8, token_ids=shared + (9, 10, 11, 12))
    manager.allocate("r2", kv_tokens=7, token_ids=shared + (91, 92, 3))
    # r2 reuses the first block only.
    assert manager.prefix_cache.hits == 1
    assert manager.used_blocks == 3  # 2 for r1 + 2 for r2 - 1 shared
    manager.check_invariants()


def test_different_prompts_do_not_collide():
    manager = make_manager(num_blocks=8, block_size=4)
    manager.allocate("r1", kv_tokens=8, token_ids=tuple(range(8)))
    manager.allocate("r2", kv_tokens=8, token_ids=tuple(range(10, 18)))
    assert manager.prefix_cache.hits == 0
    assert manager.used_blocks == 4


def test_released_cached_blocks_are_reused():
    manager = make_manager(num_blocks=8, block_size=4)
    tokens = tuple(range(8))
    manager.allocate("r1", kv_tokens=8, token_ids=tokens)
    manager.release("r1")
    assert manager.used_blocks == 0
    # A new request with the same prompt finds the blocks again.
    manager.allocate("r2", kv_tokens=8, token_ids=tokens)
    assert manager.prefix_cache.hits == 2
    assert manager.used_blocks == 2
    manager.check_invariants()


def test_disabled_cache_allocates_fresh():
    manager = make_manager(enable=False)
    manager.allocate("r1", kv_tokens=4, token_ids=(1, 2, 3, 4))
    manager.allocate("r2", kv_tokens=4, token_ids=(1, 2, 3, 4))
    assert manager.used_blocks == 2
    assert manager.prefix_cache is None


def test_no_token_ids_falls_back_to_fresh_allocation():
    manager = make_manager()
    manager.allocate("r1", kv_tokens=4)
    assert manager.prefix_cache.misses == 1
    assert manager.prefix_cache.hits == 0


def test_extend_after_shared_prefix():
    manager = make_manager(num_blocks=8, block_size=4)
    tokens = tuple(range(8))
    manager.allocate("r1", kv_tokens=8, token_ids=tokens)
    manager.allocate("r2", kv_tokens=8, token_ids=tokens)
    # Shared blocks are full at 8 tokens; the next decoded token claims a block.
    assert manager.extend("r2") is True
    assert manager.used_blocks == 3
    manager.check_invariants()


def test_release_of_shared_block_keeps_alive_for_other_owner():
    manager = make_manager(num_blocks=8, block_size=4)
    tokens = tuple(range(8))
    manager.allocate("r1", kv_tokens=8, token_ids=tokens)
    manager.allocate("r2", kv_tokens=8, token_ids=tokens)
    manager.release("r1")
    assert manager.sequence_blocks("r2") == [0, 1]
    assert manager.used_blocks == 2
    manager.release("r2")
    assert manager.used_blocks == 0
    # cached contents retained for future hits
    assert manager.blocks[0].token_ids == (0, 1, 2, 3)
    manager.check_invariants()


def test_hash_tokens_stable():
    assert hash_tokens((1, 2, 3)) == hash_tokens((1, 2, 3))
    assert hash_tokens((1, 2, 3)) != hash_tokens((3, 2, 1))


def test_metrics_sink_receives_per_event_counters():
    """Hits/misses flow to a metrics sink per event (no cross-engine deltas)."""
    metrics = EngineMetrics()  # own registry: isolated from the global singleton
    manager = BlockManager(
        num_blocks=8, block_size=4, enable_prefix_cache=True, prefix_metrics=metrics
    )
    tokens = tuple(range(8))
    manager.allocate("r1", kv_tokens=8, token_ids=tokens)
    manager.allocate("r2", kv_tokens=8, token_ids=tokens)
    assert metrics.registry.get_sample_value("fluxllm_prefix_cache_misses_total") == 2.0
    assert metrics.registry.get_sample_value("fluxllm_prefix_cache_hits_total") == 2.0
    assert metrics.registry.get_sample_value("fluxllm_prefix_cache_reused_tokens_total") == 8.0
    # A second manager wired to the same sink only ever increments.
    manager2 = BlockManager(
        num_blocks=8, block_size=4, enable_prefix_cache=True, prefix_metrics=metrics
    )
    manager2.allocate("r3", kv_tokens=8, token_ids=tokens)
    assert metrics.registry.get_sample_value("fluxllm_prefix_cache_misses_total") == 4.0


def test_reserved_tail_blocks_are_never_shared():
    """Blocks covering reserved tail slots (no prompt token content) stay private."""
    metrics = EngineMetrics()
    manager = BlockManager(
        num_blocks=16, block_size=4, enable_prefix_cache=True, prefix_metrics=metrics
    )
    # 4 prompt tokens + 4 reserved slots = 2 blocks; block 2 covers tail slots.
    manager.allocate("r1", kv_tokens=8, token_ids=(1, 2, 3, 4))
    manager.allocate("r2", kv_tokens=8, token_ids=(1, 2, 3, 4))
    # Only the one full prompt block is shared; the tail block is private.
    assert manager.prefix_cache.hits == 1
    assert manager.used_blocks == 3
    manager.check_invariants()
