"""Block-based KV-cache management (logical paged memory) + prefix cache.

Inspired by paged virtual memory (and documented honestly as *logical*
rather than kernel-level paging): a sequence's KV entries live in fixed-size
blocks that are allocated on demand as the sequence grows and released on
completion. Blocks are refcounted so sequences that agree on a prefix can
share block ownership in the accounting layer (see PrefixCache below).

The manager is pure bookkeeping: it never touches tensors. Its job is
capacity control and precise accounting of which sequence holds which
block, so the engine can admit/preempt sequences based on real cache
pressure. Scope note: because the HF executor stores per-sequence KV
tensors, prefix reuse here is accounting-level (fewer blocks rented,
higher effective capacity); attention-kernel-level KV sharing (true
PagedAttention) is intentionally out of scope and documented as such.

Allocation model:

- ``allocate`` reserves blocks covering a sequence's prompt at admission,
  reusing *full* cached prefix blocks by content hash when available
  (blocks covering reserved tail slots carry no prompt content and are
  never shared);
- ``extend`` consumes one token slot per decoded token, claiming a fresh
  block only when the last open block fills up (checked internally);
- admission decisions use ``can_allocate``/``can_extend`` *before* work
  happens, so the scheduler never admits a sequence the pool cannot serve;
- ``release`` frees all blocks on completion, decrementing refcounts;
  cached blocks stay registered for future prefix reuse, while blocks
  claimed without a cache match (e.g. decode growth) drop their cache
  identity so account sharing can never mix different contents.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Protocol


class CacheFullError(Exception):
    """Raised when an allocation cannot be satisfied from the free pool."""


def hash_tokens(tokens: tuple[int, ...]) -> int:
    """Stable content hash for one block of tokens (chain element)."""
    packed = b"".join((token & 0xFFFFFFFF).to_bytes(4, "little") for token in tokens)
    return zlib.crc32(packed)


@dataclass
class Block:
    """One fixed-size group of KV token slots."""

    block_id: int
    # Reference count: >0 means one or more sequences hold this block.
    refs: int = 0
    # Token ids conceptually stored in this block (first owner only). Used by
    # the prefix cache to detect identical blocks across sequences; None for
    # blocks that cover reserved tail slots instead of real prompt tokens.
    token_ids: tuple[int, ...] | None = None
    # True once the block is registered in the prefix cache; a block with
    # refs==0 and cached=True is retained for reuse by future requests.
    cached: bool = False

    @property
    def is_free(self) -> bool:
        return self.refs == 0


@dataclass
class SequenceBlocks:
    """Block table for one sequence."""

    sequence_id: str
    blocks: list[int] = field(default_factory=list)
    # Number of KV token slots this sequence currently occupies in the pool.
    kv_tokens: int = 0


class PrefixCacheMetrics(Protocol):
    """Duck-typed metrics sink for prefix-cache events.

    Satisfied by :class:`fluxllm.metrics.metrics.EngineMetrics`. Events
    flow per allocation (hit/miss), never as cross-engine snapshot deltas.
    """

    def on_prefix_hit(self, reused_tokens: int) -> None: ...

    def on_prefix_miss(self) -> None: ...


class PrefixCache:
    """Content-addressed registry of prefix blocks.

    Keys are cumulative chains of per-block content hashes, so a chain
    matches only requests whose token prefix matches block-for-block
    (full-block granularity: the trailing partial block is never shared).
    """

    def __init__(
        self,
        block_manager: BlockManager,
        metrics: PrefixCacheMetrics | None = None,
    ) -> None:
        self._manager = block_manager
        # Optional per-event metrics sink, wired by the engine.
        self._metrics = metrics
        self.registry: dict[tuple[int, ...], int] = {}
        self.hits = 0
        self.misses = 0
        self.reused_tokens = 0

    def lookup(self, chain: tuple[int, ...], block_tokens: tuple[int, ...]) -> int | None:
        block_id = self.registry.get(chain)
        if block_id is None:
            return None
        block = self._manager.blocks[block_id]
        if block.token_ids != block_tokens:
            # Stale/colliding entry (block content changed or cleared by a
            # generic claim): drop it so the chain is never mis-shared.
            del self.registry[chain]
            return None
        return block_id

    def register(self, chain: tuple[int, ...], block_id: int) -> None:
        self.registry[chain] = block_id

    def record_hit(self, reused_tokens: int) -> None:
        self.hits += 1
        self.reused_tokens += reused_tokens
        if self._metrics is not None:
            self._metrics.on_prefix_hit(reused_tokens)

    def record_miss(self) -> None:
        self.misses += 1
        if self._metrics is not None:
            self._metrics.on_prefix_miss()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def snapshot(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "reused_tokens": self.reused_tokens,
            "hit_rate": self.hit_rate,
            "registered_blocks": len(self.registry),
        }


class BlockManager:
    """Fixed-capacity logical block pool with refcounted ownership."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_cache: bool = False,
        prefix_metrics: PrefixCacheMetrics | None = None,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.block_size = block_size
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.tables: dict[str, SequenceBlocks] = {}
        self.prefix_cache: PrefixCache | None = (
            PrefixCache(self, metrics=prefix_metrics) if enable_prefix_cache else None
        )

    # -- pool accounting ----------------------------------------------------
    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def free_blocks(self) -> int:
        return sum(1 for b in self.blocks if b.is_free)

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - self.free_blocks

    @property
    def utilization(self) -> float:
        return self.used_blocks / self.num_blocks

    def blocks_needed(self, kv_tokens: int) -> int:
        return (kv_tokens + self.block_size - 1) // self.block_size

    def snapshot(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "used_blocks": self.used_blocks,
            "free_blocks": self.free_blocks,
            "utilization": self.utilization,
            "block_size": self.block_size,
        }

    # -- sequence accounting -------------------------------------------------
    def has_sequence(self, sequence_id: str) -> bool:
        return sequence_id in self.tables

    def sequence_blocks(self, sequence_id: str) -> list[int]:
        table = self.tables.get(sequence_id)
        return list(table.blocks) if table else []

    def sequence_block_count(self, sequence_id: str) -> int:
        table = self.tables.get(sequence_id)
        return len(table.blocks) if table else 0

    def sequence_kv_tokens(self, sequence_id: str) -> int | None:
        table = self.tables.get(sequence_id)
        return table.kv_tokens if table else None

    # -- capacity questions (used pre-admission, never raise) ----------------
    def can_allocate(self, kv_tokens: int) -> bool:
        """Could a new sequence's ``kv_tokens`` tokens be held right now?

        Conservative: does not factor potential prefix reuse in.
        """
        return self.free_blocks >= self.blocks_needed(kv_tokens)

    def can_extend(self, sequence_id: str, target_kv_tokens: int) -> bool:
        """Could the sequence grow to ``target_kv_tokens`` from its current table?

        For a sequence without a table this defers to :meth:`can_allocate`.
        """
        table = self.tables.get(sequence_id)
        if table is None:
            return self.can_allocate(target_kv_tokens)
        extra = self.blocks_needed(target_kv_tokens) - self.blocks_needed(table.kv_tokens)
        if extra <= 0:
            return True
        return self.free_blocks >= extra

    # -- allocation ----------------------------------------------------------
    def _take_free_block(self) -> int:
        for block in self.blocks:
            if block.is_free:
                block.refs = 1
                if block.cached:
                    # Claimed without a validated cache match (decode growth
                    # or fresh allocation): strip the cache identity so this
                    # block can never be shared against its old content chain.
                    # Stale registry entries self-clean on the next lookup.
                    block.cached = False
                    block.token_ids = None
                return block.block_id
        raise CacheFullError("no free KV blocks")

    def allocate(
        self,
        sequence_id: str,
        kv_tokens: int,
        token_ids: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Reserve blocks covering ``kv_tokens`` token slots for a sequence.

        With a prefix cache and ``token_ids`` provided, full leading blocks
        whose hashed chain matches the registry are shared via refcounts
        instead of fresh allocation (accounting-level reuse). Blocks beyond
        the provided tokens cover reserved tail slots and are never shared.
        """
        if sequence_id in self.tables:
            raise ValueError(f"sequence {sequence_id} already has a block table")
        table = SequenceBlocks(sequence_id=sequence_id, kv_tokens=kv_tokens)
        self.tables[sequence_id] = table
        cache = self.prefix_cache
        try:
            chain: tuple[int, ...] = ()
            for i in range(self.blocks_needed(kv_tokens)):
                if token_ids:
                    chunk: tuple[int, ...] | None = (
                        tuple(token_ids[i * self.block_size : (i + 1) * self.block_size]) or None
                    )
                else:
                    chunk = None
                matched = False
                if cache is not None and chunk is not None:
                    chain += (hash_tokens(chunk),)
                    shared_id = cache.lookup(chain, chunk)
                    if shared_id is not None:
                        self.blocks[shared_id].refs += 1
                        table.blocks.append(shared_id)
                        cache.record_hit(len(chunk))
                        matched = True
                if matched:
                    continue
                block_id = self._take_free_block()  # may raise; rollback below
                table.blocks.append(block_id)
                self.blocks[block_id].token_ids = chunk
                if cache is not None:
                    cache.record_miss()
                    if chunk is not None:
                        self.blocks[block_id].cached = True
                        cache.register(chain, block_id)
        except CacheFullError:
            self._rollback(table)
            raise
        return list(table.blocks)

    def _rollback(self, table: SequenceBlocks) -> None:
        """Undo a partially-committed allocation (drop one ref per held block)."""
        for block_id in table.blocks:
            block = self.blocks[block_id]
            if block.refs <= 0:
                raise RuntimeError(f"block {block_id} referenced while free")
            block.refs -= 1
        del self.tables[table.sequence_id]

    def extend(self, sequence_id: str) -> bool:
        """Consume one more token slot for ``sequence_id``.

        Returns True when a fresh block was claimed, False when the token fit
        into the current last block's remaining capacity. Raises
        :class:`CacheFullError` if a new block is needed and none is free.
        """
        table = self.tables.get(sequence_id)
        if table is None:
            raise KeyError(f"sequence {sequence_id} has no block table")
        table.kv_tokens += 1
        if not table.blocks:
            table.blocks.append(self._take_free_block())
            return True
        if table.kv_tokens <= len(table.blocks) * self.block_size:
            return False
        table.blocks.append(self._take_free_block())
        return True

    def unextend(self, sequence_id: str) -> None:
        """Shrink one token slot (bookkeeping reversion; never frees blocks)."""
        table = self.tables.get(sequence_id)
        if table is None:
            raise KeyError(f"sequence {sequence_id} has no block table")
        table.kv_tokens -= 1
        if table.kv_tokens < (len(table.blocks) - 1) * self.block_size:
            raise RuntimeError("unextend freed a block: pool accounting is inconsistent")

    def release(self, sequence_id: str) -> int:
        """Release every block a sequence holds; returns the number freed."""
        table = self.tables.pop(sequence_id, None)
        if table is None:
            return 0
        freed = 0
        for block_id in table.blocks:
            block = self.blocks[block_id]
            if block.refs <= 0:
                raise RuntimeError(f"block {block_id} referenced while free")
            block.refs -= 1
            if block.refs == 0:
                if not block.cached:
                    block.token_ids = None  # cacheable blocks keep contents
                freed += 1
        return freed

    # -- invariant checking (tests) ------------------------------------------
    def check_invariants(self) -> None:
        for block in self.blocks:
            if block.refs < 0:
                raise RuntimeError(f"block {block.block_id} refs negative")
            if block.is_free and block.token_ids is not None and not block.cached:
                raise RuntimeError(f"free un-cached block {block.block_id} retains token_ids")
            if block.cached and block.token_ids is None:
                raise RuntimeError(f"cached block {block.block_id} lost token ids")
        for table in self.tables.values():
            expected = self.blocks_needed(table.kv_tokens)
            if len(table.blocks) != expected:
                raise RuntimeError(
                    f"sequence {table.sequence_id}: kv_tokens={table.kv_tokens} "
                    f"requires {expected} blocks, holds {len(table.blocks)}"
                )
