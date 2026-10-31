"""KV-cache block manager tests: allocation, growth, release, exhaustion."""

from __future__ import annotations

import pytest

from fluxllm.cache.block_manager import (
    BlockManager,
    CacheFullError,
)


def make_manager(num_blocks=8, block_size=4):
    return BlockManager(num_blocks=num_blocks, block_size=block_size)


def test_allocate_reserves_blocks_for_prompt():
    blocks = make_manager(num_blocks=8, block_size=4)
    ids = blocks.allocate("r1", kv_tokens=5, token_ids=(1, 2, 3, 4, 5))
    assert len(ids) == 2  # 5 tokens need 2 blocks
    assert blocks.used_blocks == 2
    assert blocks.sequence_kv_tokens("r1") == 5
    blocks.check_invariants()


def test_extend_grows_into_open_block_then_claims_new_one():
    blocks = make_manager(num_blocks=8, block_size=4)
    blocks.allocate("r1", kv_tokens=5)
    assert blocks.extend("r1") is False  # 6th token fits in block 2
    assert blocks.sequence_kv_tokens("r1") == 6
    assert blocks.used_blocks == 2
    # kv 7, 8 fill block 2; token 9 opens block 3
    assert blocks.extend("r1") is False
    assert blocks.extend("r1") is False
    assert blocks.extend("r1") is True
    assert blocks.used_blocks == 3
    blocks.check_invariants()


def test_exhaustion_raises_and_leaves_pool_consistent():
    blocks = make_manager(num_blocks=2, block_size=4)
    blocks.allocate("r1", kv_tokens=4)
    with pytest.raises(CacheFullError):
        blocks.allocate("r2", kv_tokens=8)
    assert not blocks.has_sequence("r2")  # rolled back cleanly
    assert blocks.free_blocks == 1
    blocks.check_invariants()


def test_release_frees_all_blocks():
    blocks = make_manager(num_blocks=8, block_size=4)
    blocks.allocate("r1", kv_tokens=5)
    assert blocks.used_blocks == 2
    for _ in range(4):  # kv 6..9; token 9 opens a third block
        blocks.extend("r1")
    assert blocks.used_blocks == 3
    freed = blocks.release("r1")
    assert freed == 3
    assert blocks.free_blocks == 8
    assert not blocks.has_sequence("r1")
    blocks.check_invariants()


def test_capacity_questions():
    blocks = make_manager(num_blocks=4, block_size=4)
    assert blocks.can_allocate(kv_tokens=16) is True
    assert blocks.can_allocate(kv_tokens=17) is False
    blocks.allocate("r1", kv_tokens=4)
    # kv 5..8: opens one more block (3 free available)
    assert blocks.can_extend("r1", target_kv_tokens=8) is True
    # kv 9: needs a third block; 2 of 3 free remain
    assert blocks.can_extend("r1", target_kv_tokens=9) is True
    # kv 17: would need 5 blocks total but the pool only has 4
    assert blocks.can_extend("r1", target_kv_tokens=17) is False
    # no-table sequences defer to can_allocate (3 free blocks remain)
    assert blocks.can_extend("ghost", target_kv_tokens=12) is True
    assert blocks.can_extend("ghost", target_kv_tokens=13) is False


def test_double_allocate_rejected():
    blocks = make_manager()
    blocks.allocate("r1", kv_tokens=4)
    with pytest.raises(ValueError):
        blocks.allocate("r1", kv_tokens=4)


def test_extend_without_table_raises():
    blocks = make_manager()
    with pytest.raises(KeyError):
        blocks.extend("nope")


def test_release_without_table_is_noop():
    assert make_manager().release("nope") == 0


def test_utilization_tracking():
    blocks = make_manager(num_blocks=10, block_size=4)
    assert blocks.utilization == 0.0
    blocks.allocate("r1", kv_tokens=8)
    assert blocks.utilization == pytest.approx(0.2)
    blocks.allocate("r2", kv_tokens=12)
    assert blocks.utilization == pytest.approx(0.5)


def test_multiple_sequences_share_pool_and_release_independently():
    blocks = make_manager(num_blocks=6, block_size=4)
    blocks.allocate("a", kv_tokens=4)
    blocks.allocate("b", kv_tokens=4)
    assert blocks.free_blocks == 4
    blocks.release("a")
    assert blocks.free_blocks == 5
    assert blocks.sequence_block_count("b") == 1
    blocks.check_invariants()


def test_snapshot_shape():
    snap = make_manager(num_blocks=5, block_size=2).snapshot()
    assert snap == {
        "num_blocks": 5,
        "used_blocks": 0,
        "free_blocks": 5,
        "utilization": 0.0,
        "block_size": 2,
    }
