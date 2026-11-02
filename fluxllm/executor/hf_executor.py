"""Hugging Face causal LM executor (PyTorch).

Runs one sequence per forward call. A real per-sequence KV cache is kept via
transformers' ``DynamicCache`` so prompts are never recomputed during
decoding.

Scope note (important): decode forwards here are per-sequence, NOT batched
through a paged attention kernel. Production engines batch the attention
computation across sequences within a step; FluxLLM batches at the
scheduling level. This keeps the executor simple, correct, and portable
across CPU/CUDA, and is documented as a deliberate limitation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from fluxllm.executor.base import SequenceState, StepOutput
from fluxllm.executor.device import detect_device, resolve_dtype

if TYPE_CHECKING:
    import torch

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
        ids = self.torch.tensor([list(token_ids)], dtype=self.torch.long)
        return self._forward(request_id, ids, None)

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

        Callers must have validated the requests are not already prefilled;
        this method validates all rows before any state is touched, so a
        failure leaves no partial registrations behind.
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
        bsz = len(prompts)
        lengths = [len(tokens) for _, tokens in prompts]
        max_prompt = max(lengths)
        pad_id = int(
            getattr(self.tokenizer, "pad_token_id", None)
            or getattr(self.tokenizer, "eos_token_id", None)
            or 0
        )
        ids = torch.full((bsz, max_prompt), pad_id, dtype=torch.long)
        attention = torch.zeros((bsz, max_prompt), dtype=torch.long)
        position_ids = torch.zeros((bsz, max_prompt), dtype=torch.long)
        for row, (_, tokens) in enumerate(prompts):
            n = len(tokens)
            ids[row, max_prompt - n :] = torch.tensor(list(tokens), dtype=torch.long)
            attention[row, max_prompt - n :] = 1
            position_ids[row, max_prompt - n :] = torch.arange(n)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=ids.to(self.device),
                attention_mask=attention.to(self.device),
                position_ids=position_ids.to(self.device),
                past_key_values=None,
                use_cache=True,
            )
        past = outputs.past_key_values
        results: list[StepOutput] = []
        for row, (request_id, _tokens) in enumerate(prompts):
            n = lengths[row]
            entry = {
                "hf_cache": self._row_cache(past, row, max_prompt - n, n),
                "num_tokens": n,
            }
            self._caches[request_id] = entry
            state = SequenceState(request_id=request_id, num_tokens=n, kv=entry)
            results.append(
                StepOutput(
                    logits=outputs.logits[row, -1, :].float().detach().cpu().numpy(),
                    state=state,
                    context_exceeded=n >= self._max_context_len,
                )
            )
        return results

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
