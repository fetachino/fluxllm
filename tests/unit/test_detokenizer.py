"""Incremental detokenizer tests: stable deltas and partial UTF-8."""

from __future__ import annotations

from fluxllm.engine.detokenizer import IncrementalDetokenizer


def test_deltas_reconstruct_full_text():
    decoder = lambda ids: "".join(chr(ord("a") + i) for i in ids)  # noqa: E731
    detok = IncrementalDetokenizer(decoder)
    pieces = [detok.update(i) for i in range(5)]
    assert "".join(pieces) == "abcde"
    assert detok.finalize() == ""
    assert detok.text == "abcde"
    assert detok.token_ids == [0, 1, 2, 3, 4]


def test_update_many_is_equivalent_to_per_token_updates():
    decoder = lambda ids: "".join(str(i) for i in ids)  # noqa: E731
    detok = IncrementalDetokenizer(decoder)
    assert detok.update_many([1, 2]) == "12"
    assert detok.update(3) == "3"
    assert detok.finalize() == ""


def test_partial_utf8_is_held_back_until_complete():
    # Each token is a raw byte; decode with replacement exactly like a
    # byte-level BPE tokenizer. "é" is 0xC3 0xA9.
    def decoder(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    detok = IncrementalDetokenizer(decoder)
    assert detok.update(0xC3) == ""  # incomplete sequence held back
    assert detok.update(0xA9) == "é"
    assert detok.finalize() == ""
    assert "\ufffd" not in detok.text


def test_finalize_flushes_an_unterminated_partial_character():
    def decoder(ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")

    detok = IncrementalDetokenizer(decoder)
    assert detok.update(0xC3) == ""
    assert detok.finalize() == "\ufffd"


def test_revision_never_retracts_emitted_text():
    def decoder(ids: list[int]) -> str:
        # First token decodes to "a"; adding the second token revises the
        # whole string. The detokenizer must not emit a contradictory delta.
        return "a" if len(ids) == 1 else "b"

    detok = IncrementalDetokenizer(decoder)
    assert detok.update(1) == "a"
    assert detok.update(2) == ""
    # finalize emits the divergent tail after the common prefix.
    assert detok.finalize() == "b"


def test_finalize_is_idempotent():
    detok = IncrementalDetokenizer(lambda ids: "done")
    assert detok.update(1) == "done"
    assert detok.finalize() == ""
    assert detok.finalize() == ""
