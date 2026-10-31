"""Deterministic fallback tokenization for engineering-simulation mode.

Used only when the engine runs the :class:`DeterministicExecutor` (unit
tests, scheduler simulation, and the clearly-labeled demo mode). Token ids
are stable CRC32 hashes of whitespace-separated words; detokenization is a
labeled placeholder because the mapping is not invertible. This mode never
poses as real model output.
"""

from __future__ import annotations

import zlib


def simple_tokenize(text: str, vocab_size: int) -> list[int]:
    """Map words to stable ids in [1, vocab_size) (0 reserved as padding)."""
    if vocab_size < 2:
        raise ValueError("vocab_size must be >= 2")
    ids = [(zlib.crc32(word.encode("utf-8")) % (vocab_size - 1)) + 1 for word in text.split()]
    return ids if ids else [1]


def simple_detokenize(token_ids: list[int]) -> str:
    """Labeled placeholder rendering of simulated tokens."""
    return " ".join(f"[sim {token_id}]" for token_id in token_ids)
