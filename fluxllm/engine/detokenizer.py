"""Incremental detokenization for streaming responses.

Token-level streaming cannot naively decode each accumulated token list and
emit the string suffix: byte-level BPE tokenizers can emit a partial UTF-8
sequence, which decodes to U+FFFD until the following token completes it.
Emitting that replacement character would corrupt the client-visible text
and force a retraction later.

:class:`IncrementalDetokenizer` keeps the accumulated token ids and the text
already emitted. Each update decodes the sequence, trims any *trailing*
replacement character (an incomplete multi-byte sequence), and returns only
the new stable suffix. :meth:`finalize` flushes whatever is left when the
request terminates, including a trailing replacement character that never
completed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def _is_partial_char(ch: str) -> bool:
    """True for code points that indicate an incomplete UTF-8 sequence."""
    return ch == "\ufffd" or "\ud800" <= ch <= "\udfff"


def _stable_text(text: str) -> str:
    """Drop trailing code points that a later token could still complete."""
    end = len(text)
    while end > 0 and _is_partial_char(text[end - 1]):
        end -= 1
    return text[:end]


class IncrementalDetokenizer:
    """Turn a token stream into stable, monotonic text deltas."""

    def __init__(self, decode: Callable[[list[int]], str]) -> None:
        self._decode = decode
        self._ids: list[int] = []
        self._emitted = ""

    @property
    def token_ids(self) -> list[int]:
        return list(self._ids)

    @property
    def text(self) -> str:
        """Text emitted so far (excludes held-back partial characters)."""
        return self._emitted

    def update(self, token_id: int) -> str:
        """Append one token and return the newly stable text (may be "")."""
        self._ids.append(int(token_id))
        return self._refresh()

    def update_many(self, token_ids: Sequence[int]) -> str:
        """Append several tokens and return the newly stable text."""
        self._ids.extend(int(t) for t in token_ids)
        return self._refresh()

    def _refresh(self) -> str:
        stable = _stable_text(self._decode(self._ids))
        if not stable.startswith(self._emitted):
            # The decoder revised already-emitted text (rare: cleanup rules
            # that depend on later tokens). Never retract bytes a client has
            # already seen; hold until the decode converges again.
            return ""
        delta = stable[len(self._emitted) :]
        self._emitted = stable
        return delta

    def finalize(self) -> str:
        """Flush remaining text when generation ends (idempotent)."""
        final = self._decode(self._ids)
        if final.startswith(self._emitted):
            delta = final[len(self._emitted) :]
        else:
            # The final decode diverged from what was emitted. Emit the
            # divergent tail rather than silently dropping it; the client
            # cannot un-see the earlier prefix.
            common = 0
            limit = min(len(self._emitted), len(final))
            while common < limit and self._emitted[common] == final[common]:
                common += 1
            delta = final[common:]
        self._emitted = final
        return delta


__all__ = ["IncrementalDetokenizer"]
