"""Block-based KV-cache management (logical paged memory).

Inspired by paged virtual memory (and documented honestly as *logical*
rather than kernel-level paging): a sequence's KV entries live in fixed-size
blocks that are allocated on demand as the sequence grows and released on
completion. Blocks are refcounted so a prefix-cache layer can share blocks
across sequences that agree on a prefix.

The manager is pure bookkeeping: it never touches tensors. Its job is
capacity control and precise accounting of which sequence owns which block,
so the engine can admit/preempt sequences based on real cache pressure.

Allocation model:

- ``allocate`` reserves blocks covering a sequence's prompt at admission;
- ``extend`` consumes one token slot per decoded token, claiming a fresh
  block only when the last open block fills up (checked internally);
- admission decisions use ``can_allocate``/``can_extend`` *before* work
  happens, so the scheduler never admits a sequence the pool cannot serve;
- ``release`` frees all blocks on completion, decrementing refcounts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class CacheFullError(Exception):
    """Raised when an allocation cannot be satisfied from the free pool."""


@dataclass
class Block:
    """One fixed-size group of KV token slots."""

    block_id: int
    # Reference count: >0 means one or more sequences hold this block.
    refs: int = 0
    # Token ids conceptually stored in this block (first owner only). Used by
    # the prefix cache to detect identical blocks across sequences.
    token_ids: tuple[int, ...] | None = None

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


class BlockManager:
    """Fixed-capacity logical block pool with refcounted ownership."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.block_size = block_size
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.tables: dict[str, SequenceBlocks] = {}

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
        """Could a new sequence's ``kv_tokens`` tokens be held right now?"""
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
                return block.block_id
        raise CacheFullError("no free KV blocks")

    def allocate(
        self,
        sequence_id: str,
        kv_tokens: int,
        token_ids: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Reserve blocks covering ``kv_tokens`` token slots for a sequence."""
        if sequence_id in self.tables:
            raise ValueError(f"sequence {sequence_id} already has a block table")
        table = SequenceBlocks(sequence_id=sequence_id, kv_tokens=kv_tokens)
        self.tables[sequence_id] = table
        try:
            for i in range(self.blocks_needed(kv_tokens)):
                block_id = self._take_free_block()  # may raise; cleanup below
                table.blocks.append(block_id)
                self.blocks[block_id].token_ids = (
                    tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
                    if token_ids
                    else None
                )
        except CacheFullError:
            self._rollback(table)
            raise
        return list(table.blocks)

    def _rollback(self, table: SequenceBlocks) -> None:
        for block_id in table.blocks:
            self.blocks[block_id].refs = 0
            self.blocks[block_id].token_ids = None
        del self.tables[table.sequence_id]

    def acquire(self, sequence_id: str, block_id: int, kv_tokens: int) -> None:
        """Attach an existing (shared) block to a sequence; bumps its refcount."""
        block = self.blocks[block_id]
        block.refs += 1
        table = self.tables.get(sequence_id)
        if table is None:
            table = SequenceBlocks(sequence_id=sequence_id, kv_tokens=kv_tokens)
            self.tables[sequence_id] = table
        else:
            table.kv_tokens = kv_tokens
        table.blocks.append(block_id)

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
        """Shrink one token slot (used when reverting over-admission)."""
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
                block.token_ids = None
                freed += 1
        return freed

    # -- invariant checking (tests) ------------------------------------------
    def check_invariants(self) -> None:
        for block in self.blocks:
            if block.is_free and block.token_ids is not None:
                raise RuntimeError(f"free block {block.block_id} retains token_ids")
            if block.refs < 0:
                raise RuntimeError(f"block {block.block_id} refs negative")
        for table in self.tables.values():
            expected = self.blocks_needed(table.kv_tokens)
            if len(table.blocks) != expected:
                raise RuntimeError(
                    f"sequence {table.sequence_id}: kv_tokens={table.kv_tokens} "
                    f"requires {expected} blocks, holds {len(table.blocks)}"
                )
