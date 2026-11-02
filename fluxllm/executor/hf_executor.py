"""Hugging Face causal LM executor (PyTorch).

Runs one sequence per forward call. A real per-sequence KV cache is kept via
transformers' ``DynamicCache`` so prompts are never recomputed during
decoding.

Scope note (important): decode forwards are per-sequence unless the engine
batches them (``decode_batch``); even then attention runs through the
model's own kernel, not a paged attention kernel. FluxLLM's scheduler still
owns batching decisions, and the executor documents what it does and does
not batch.

Prefix reuse: completed prefills are snapshotted (cloned KV + next-token
logits) in a small bounded LRU. A later prompt that shares a cached prefix
clones that KV and only forwards the remaining tokens — or returns the
stored logits outright on an exact match. This is executor-level compute
reuse, distinct from the block manager's capacity-level sharing.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from fluxllm.executor.base import SequenceState, StepOutput
from fluxllm.executor.device import detect_device, resolve_dtype

if TYPE_CHECKING:
    import torch

    from fluxllm.metrics.metrics import EngineMetrics

logger = logging.getLogger(__name__)


class HFModelExecutor:
    """Executes a Hugging Face causal language model one sequence at a time."""

    def __init__(
        self,
        model_name: str,
        dtype: str = "float32",
        device: str = "auto",
        trust_remote_code: bool = False,
        max_context_len: int | None = None,
        enable_prefix_compute_reuse: bool = True,
        prefix_kv_entries: int = 8,
        prefix_kv_max_tokens: int = 8192,
        metrics: EngineMetrics | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device_name = detect_device(device)
        self.device = torch.device(self.device_name)
        self.dtype = resolve_dtype(dtype, self.device_name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
            trust_remote_code=trust_remote_code,
        )
        # transformers 5.x wraps Module.to; its typing stubs resolve poorly
        # across versions and platforms, so route through Any explicitly.
        cast(Any, self.model).to(self.device)
        self.model.eval()

        config = self.model.config
        self._vocab_size = int(config.vocab_size)
        self._eos_token_id = config.eos_token_id if isinstance(config.eos_token_id, int) else None
        self._max_context_len = (
            max_context_len
            if max_context_len is not None
            else int(getattr(config, "max_position_embeddings", 4096))
        )
        # request_id -> {"hf_cache": Cache, "num_tokens": int}
        self._caches: dict[str, dict] = {}

        # Executor-level prefix reuse: token-prefix -> (cloned KV, next logits).
        self._enable_prefix_reuse = enable_prefix_compute_reuse
        self._prefix_max_entries = max(1, prefix_kv_entries)
        self._prefix_max_tokens = max(1, prefix_kv_max_tokens)
        self._prefix_store: OrderedDict[tuple[int, ...], tuple[Any, np.ndarray]] = OrderedDict()
        self._prefix_store_tokens = 0
        self._metrics = metrics
        # Cumulative tokens whose KV was reused instead of recomputed.
        self.prefix_reused_tokens = 0

        logger.info(
            "loaded model %s on %s (dtype=%s, vocab=%d, eos=%s, max_context=%d)",
            model_name,
            self.device_name,
            dtype,
            self._vocab_size,
            self._eos_token_id,
            self._max_context_len,
        )

    # -- properties ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    @property
    def max_context_len(self) -> int:
        return self._max_context_len

    # -- tokenization helpers (engine-level convenience) ---------------------
    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=True))

    def decode_text(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))

    # -- lifecycle ----------------------------------------------------------
    def release(self, request_id: str) -> None:
        self._caches.pop(request_id, None)

    # -- forward ------------------------------------------------------------
    def _forward(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        entry: dict | None,
    ) -> StepOutput:
        torch = self.torch
        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                past_key_values=entry["hf_cache"] if entry is not None else None,
                use_cache=True,
            )
        if entry is None:
            entry = {}
        entry["hf_cache"] = outputs.past_key_values
        # KV length after consuming input_ids; for prefill this equals the
        # prompt length, for decode it grows by one per step.
        entry["num_tokens"] = int(outputs.past_key_values.get_seq_length())
        self._caches[request_id] = entry
        state = SequenceState(request_id=request_id, num_tokens=entry["num_tokens"], kv=entry)
        return StepOutput(
            logits=outputs.logits[0, -1, :].float().detach().cpu().numpy(),
            state=state,
            context_exceeded=entry["num_tokens"] >= self._max_context_len,
        )

    def prefill(self, request_id: str, token_ids: Sequence[int]) -> StepOutput:
        if not token_ids:
            raise ValueError("prefill requires at least one token")
        if len(token_ids) > self._max_context_len:
            raise ValueError("prompt exceeds model context length")
        if request_id in self._caches:
            raise RuntimeError(f"sequence {request_id} already prefilled")
        try:
            reused_out, reused_tokens = self._try_prefix_reuse(request_id, token_ids)
            if reused_out is not None:
                self._apply_reuse_metrics(reused_tokens)
                self._remember_prefix(
                    token_ids, self._caches[request_id]["hf_cache"], reused_out.logits
                )
                return reused_out
            ids = self.torch.tensor([list(token_ids)], dtype=self.torch.long)
            out = self._forward(request_id, ids, None)
            self._remember_prefix(token_ids, self._caches[request_id]["hf_cache"], out.logits)
            return out
        except Exception:
            self._caches.pop(request_id, None)
            raise

    # -- executor-level prefix reuse -----------------------------------------
    #
    # A prefill snapshot is the KV for a token prefix plus the logits that
    # predict the token after it. Reuse cases, all producing the exact logits
    # a cold prefill would produce:
    #
    # * exact match   -> return the stored logits (zero forward work);
    # * shorter key   -> clone the KV and forward only the remaining tokens;
    # * covering key  -> the prompt is a prefix of a cached longer prompt;
    #                    crop the KV to len-1 and forward the last token.
    #
    # Snapshots are cloned, never shared: decoding mutates a cache in place.

    def _clone_cache(self, cache: Any, length: int | None = None) -> Any:
        from transformers import DynamicCache

        clone = DynamicCache()
        for layer_idx in range(self._num_layers()):
            layer = cache.layers[layer_idx]
            if length is None:
                k = layer.keys.clone()
                v = layer.values.clone()
            else:
                k = layer.keys[:, :, :length, :].clone()
                v = layer.values[:, :, :length, :].clone()
            clone.update(k, v, layer_idx)
        return clone

    def _remember_prefix(self, tokens: Sequence[int], cache: Any, logits: np.ndarray) -> None:
        if not self._enable_prefix_reuse:
            return
        key = tuple(int(t) for t in tokens)
        if not key:
            return
        if key in self._prefix_store:
            self._prefix_store_tokens -= len(key)
            self._prefix_store.pop(key)
        self._prefix_store[key] = (self._clone_cache(cache), logits.copy())
        self._prefix_store_tokens += len(key)
        while (
            len(self._prefix_store) > self._prefix_max_entries
            or self._prefix_store_tokens > self._prefix_max_tokens
        ) and len(self._prefix_store) > 1:
            old_key, _ = self._prefix_store.popitem(last=False)
            self._prefix_store_tokens -= len(old_key)

    def _find_reusable_prefix(self, tokens: Sequence[int]) -> tuple[int, Any, np.ndarray] | None:
        """Longest cached key that is a prefix of ``tokens``."""
        seq = tuple(int(t) for t in tokens)
        best: tuple[int, Any, np.ndarray] | None = None
        for key, (cache, logits) in self._prefix_store.items():
            if (
                len(key) <= len(seq)
                and seq[: len(key)] == key
                and (best is None or len(key) > best[0])
            ):
                best = (len(key), cache, logits)
        return best

    def _find_covering_prefix(self, tokens: Sequence[int]) -> tuple[int, Any] | None:
        """Shortest cached key that extends ``tokens`` (prompt is a prefix)."""
        seq = tuple(int(t) for t in tokens)
        best: tuple[int, Any] | None = None
        for key, (cache, _logits) in self._prefix_store.items():
            if (
                len(key) > len(seq)
                and key[: len(seq)] == seq
                and (best is None or len(key) < best[0])
            ):
                best = (len(key), cache)
        return best

    def _try_prefix_reuse(
        self, request_id: str, token_ids: Sequence[int]
    ) -> tuple[StepOutput | None, int]:
        """Register ``request_id`` from a cached prefix if possible.

        Returns ``(output, reused_tokens)``; ``(None, 0)`` means a cold
        prefill is required. The cache entry is registered here so partial
        reuse can forward the remaining tokens; callers must roll the entry
        back if their enclosing operation fails.
        """
        if not self._enable_prefix_reuse or not self._prefix_store:
            return None, 0
        tokens = [int(t) for t in token_ids]
        try:
            found = self._find_reusable_prefix(tokens)
            if found is not None:
                matched, cache, logits = found
                cloned = self._clone_cache(cache)
                entry = {"hf_cache": cloned, "num_tokens": len(tokens)}
                self._caches[request_id] = entry
                if matched == len(tokens):
                    state = SequenceState(request_id=request_id, num_tokens=len(tokens), kv=entry)
                    return (
                        StepOutput(
                            logits=logits.copy(),
                            state=state,
                            context_exceeded=len(tokens) >= self._max_context_len,
                        ),
                        matched,
                    )
                rest = self.torch.tensor([tokens[matched:]], dtype=self.torch.long)
                out = self._forward(request_id, rest, entry)
                return out, matched
            covering = self._find_covering_prefix(tokens)
            if covering is not None and len(tokens) > 1:
                _, cache = covering
                cropped = self._clone_cache(cache, length=len(tokens) - 1)
                entry = {"hf_cache": cropped, "num_tokens": len(tokens) - 1}
                self._caches[request_id] = entry
                ids = self.torch.tensor([[tokens[-1]]], dtype=self.torch.long)
                out = self._forward(request_id, ids, entry)
                return out, len(tokens) - 1
            return None, 0
        except Exception:
            # A failed reuse must not leave a half-registered sequence.
            self._caches.pop(request_id, None)
            raise

    def _apply_reuse_metrics(self, reused_tokens: int) -> None:
        if reused_tokens <= 0:
            return
        self.prefix_reused_tokens += reused_tokens
        if self._metrics is not None:
            self._metrics.on_prefix_compute_reuse(reused_tokens)

    def decode(self, state: SequenceState, last_token: int) -> StepOutput:
        entry = self._caches.get(state.request_id)
        if entry is None or entry["num_tokens"] != state.num_tokens:
            raise RuntimeError(f"decode for unknown/stale sequence {state.request_id}")
        ids = self.torch.tensor([[last_token]], dtype=self.torch.long)
        return self._forward(state.request_id, ids, entry)

    # -- batched execution ----------------------------------------------------
    #
    # Sequences in a scheduler iteration are packed into ONE forward per
    # phase. KV tensors are gathered from per-sequence ``DynamicCache``s into
    # a batched, left-padded batch (each row's real tokens occupy the
    # trailing columns), position ids and attention masks are built per row,
    # and the resulting per-row KV is sliced back out. Left-padding keeps the
    # new token's append column identical for every row, so transformers'
    # ``DynamicCache.update`` (which appends at a shared ``cache_position``)
    # stays correct. Padded columns are zeroed in K/V and excluded by the
    # attention mask, so they cannot influence any row's attention.

    def _num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def _entry_for(self, state: SequenceState) -> dict:
        entry = self._caches.get(state.request_id)
        if entry is None or entry["num_tokens"] != state.num_tokens:
            raise RuntimeError(f"decode for unknown/stale sequence {state.request_id}")
        return entry

    def _row_cache(self, batched_cache: Any, row: int, start: int, length: int) -> Any:
        """Slice one row's KV from a batched cache into a fresh per-seq cache."""
        from transformers import DynamicCache

        cache = DynamicCache()
        for layer_idx in range(self._num_layers()):
            layer = batched_cache.layers[layer_idx]
            k = layer.keys[row : row + 1, :, start : start + length, :].clone()
            v = layer.values[row : row + 1, :, start : start + length, :].clone()
            cache.update(k, v, layer_idx)
        return cache

    def prefill_batch(self, prompts: Sequence[tuple[str, Sequence[int]]]) -> list[StepOutput]:
        """Prefill several prompts in one forward (left-padded).

        Rows whose prompt shares a cached prefix are served from that KV
        (no forward, or only the uncached tail); the remaining rows share
        one left-padded forward. Callers must have validated the requests
        are not already prefilled; this method validates all rows before any
        state is touched and rolls back every registration if any step
        fails, so a failure leaves no partial state behind.
        """
        torch = self.torch
        if not prompts:
            return []
        for request_id, tokens in prompts:
            if not tokens:
                raise ValueError("prefill requires at least one token")
            if len(tokens) > self._max_context_len:
                raise ValueError("prompt exceeds model context length")
            if request_id in self._caches:
                raise RuntimeError(f"sequence {request_id} already prefilled")

        results: list[StepOutput | None] = [None] * len(prompts)
        cold: list[int] = []
        reused: list[tuple[int, int]] = []  # (row index, reused tokens)
        registered: list[str] = []
        try:
            for row, (request_id, tokens) in enumerate(prompts):
                out, reused_tokens = self._try_prefix_reuse(request_id, tokens)
                if out is None:
                    cold.append(row)
                    continue
                results[row] = out
                registered.append(request_id)
                if reused_tokens:
                    reused.append((row, reused_tokens))
                self._remember_prefix(tokens, self._caches[request_id]["hf_cache"], out.logits)
            if cold:
                cold_prompts = [prompts[i] for i in cold]
                bsz = len(cold_prompts)
                lengths = [len(tokens) for _, tokens in cold_prompts]
                max_prompt = max(lengths)
                pad_id = int(
                    getattr(self.tokenizer, "pad_token_id", None)
                    or getattr(self.tokenizer, "eos_token_id", None)
                    or 0
                )
                ids = torch.full((bsz, max_prompt), pad_id, dtype=torch.long)
                attention = torch.zeros((bsz, max_prompt), dtype=torch.long)
                position_ids = torch.zeros((bsz, max_prompt), dtype=torch.long)
                for batch_row, (_, tokens) in enumerate(cold_prompts):
                    n = len(tokens)
                    ids[batch_row, max_prompt - n :] = torch.tensor(list(tokens), dtype=torch.long)
                    attention[batch_row, max_prompt - n :] = 1
                    position_ids[batch_row, max_prompt - n :] = torch.arange(n)
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=ids.to(self.device),
                        attention_mask=attention.to(self.device),
                        position_ids=position_ids.to(self.device),
                        past_key_values=None,
                        use_cache=True,
                    )
                past = outputs.past_key_values
                for batch_row, row in enumerate(cold):
                    request_id, tokens = cold_prompts[batch_row]
                    n = lengths[batch_row]
                    entry = {
                        "hf_cache": self._row_cache(past, batch_row, max_prompt - n, n),
                        "num_tokens": n,
                    }
                    self._caches[request_id] = entry
                    registered.append(request_id)
                    state = SequenceState(request_id=request_id, num_tokens=n, kv=entry)
                    out = StepOutput(
                        logits=outputs.logits[batch_row, -1, :].float().detach().cpu().numpy(),
                        state=state,
                        context_exceeded=n >= self._max_context_len,
                    )
                    results[row] = out
                    self._remember_prefix(tokens, entry["hf_cache"], out.logits)
        except Exception:
            for request_id in registered:
                self._caches.pop(request_id, None)
            raise

        for _row, reused_tokens in reused:
            self._apply_reuse_metrics(reused_tokens)
        return cast("list[StepOutput]", results)

    def decode_batch(self, batch: Sequence[tuple[SequenceState, int]]) -> list[StepOutput]:
        """Decode one token for several sequences in a single forward.

        Every (state, last_token) row advances by exactly one position, the
        same contract as N sequential :meth:`decode` calls; entries are only
        written back after the whole forward succeeds, so a failure leaves
        all per-sequence states untouched and callers can fall back to the
        sequential path.
        """
        torch = self.torch
        from transformers import DynamicCache

        if not batch:
            return []
        entries = [self._entry_for(state) for state, _ in batch]
        bsz = len(batch)
        lengths = [entry["num_tokens"] for entry in entries]
        max_len = max(lengths)
        k0 = entries[0]["hf_cache"].layers[0].keys
        dtype = k0.dtype
        kv_heads, head_dim = int(k0.shape[1]), int(k0.shape[3])
        num_layers = self._num_layers()
        batched_cache = DynamicCache()
        # Each layer gets its own padded gather: per-row real KV occupies the
        # trailing columns and pad columns stay zero (attention-masked).
        for layer_idx in range(num_layers):
            padded_k = torch.zeros((bsz, kv_heads, max_len, head_dim), dtype=dtype)
            padded_v = torch.zeros_like(padded_k)
            for row, entry in enumerate(entries):
                n = entry["num_tokens"]
                layer = entry["hf_cache"].layers[layer_idx]
                padded_k[row, :, max_len - n :, :] = layer.keys[0]
                padded_v[row, :, max_len - n :, :] = layer.values[0]
            batched_cache.update(padded_k, padded_v, layer_idx)
        ids = torch.tensor([[last_token] for _, last_token in batch], dtype=torch.long)
        attention = torch.zeros((bsz, max_len + 1), dtype=torch.long)
        positions = torch.zeros((bsz, 1), dtype=torch.long)
        for row, entry in enumerate(entries):
            n = entry["num_tokens"]
            # Real past tokens (trailing) plus the column of the new query.
            attention[row, max_len - n :] = 1
            positions[row, 0] = n
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids.to(self.device),
                attention_mask=attention.to(self.device),
                position_ids=positions.to(self.device),
                past_key_values=batched_cache,
                use_cache=True,
            )
        past = outputs.past_key_values
        results: list[StepOutput] = []
        for row, (state, _last_token) in enumerate(batch):
            entry = entries[row]
            new_len = entry["num_tokens"] + 1
            entry["hf_cache"] = self._row_cache(past, row, max_len - new_len + 1, new_len)
            entry["num_tokens"] = new_len
            self._caches[state.request_id] = entry
            new_state = SequenceState(request_id=state.request_id, num_tokens=new_len, kv=entry)
            results.append(
                StepOutput(
                    logits=outputs.logits[row, -1, :].float().detach().cpu().numpy(),
                    state=new_state,
                    context_exceeded=new_len >= self._max_context_len,
                )
            )
        return results
