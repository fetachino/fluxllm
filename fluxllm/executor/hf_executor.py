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
