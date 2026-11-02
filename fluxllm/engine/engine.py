"""FluxLLM engine: an async continuous-batching generation loop.

One engine coroutine owns all model work. Each iteration:

1. the scheduler plans admissions (prefills) against queue/batch/token/KV
   budgets and returns the currently decodable set;
2. newly admitted requests run their prefill forward, producing the
   first token, which is streamed immediately (TTFT measurement point);
3. every decoding sequence samples from its pending logits, each sampled
   token is streamed as soon as it is chosen (decorated TTFT and
   inter-token latency), then fed back through one decode forward;
4. pool exhaustion rolls a step back and preempts the offending
   sequence (KV recompute on re-admission, vLLM-style);
5. completions, cancellations, and failures release scheduler, executor,
   and runtime resources.

Model forwards run in a worker thread so the event loop stays responsive
to submissions and cancellations during CPU-bound inference.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, cast

import numpy as np

from fluxllm.cache.block_manager import BlockManager
from fluxllm.engine.lifecycle import RequestState
from fluxllm.engine.request import CompletionReason, GenerationRequest
from fluxllm.executor.base import ModelExecutor, SequenceState, StepOutput
from fluxllm.metrics.metrics import EngineMetrics, global_metrics
from fluxllm.sampling.config import SamplingParams
from fluxllm.sampling.sampler import Sampler
from fluxllm.scheduler.policy import SchedulerLimits
from fluxllm.scheduler.scheduler import Scheduler

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fluxllm.config.settings import EngineSettings

logger = logging.getLogger(__name__)

_DONE = object()  # stream sentinel


class EngineFullError(RuntimeError):
    """Raised when a request is rejected because the queue is full."""


@dataclass
class StreamEvent:
    """A single event on a request's token stream."""

    type: str  # "token" | "status"
    token_id: int | None = None
    state: str | None = None  # request state name (status events)
    reason: str | None = None  # completion reason (terminal status events)
    meta: dict = field(default_factory=dict)


@dataclass
class RequestHandle:
    """Client-facing handle for one submitted request."""

    engine: FluxEngine
    request: GenerationRequest
    queue: asyncio.Queue

    async def stream(self) -> AsyncIterator[StreamEvent]:
        """Yield stream events until the request reaches a terminal state."""
        while True:
            event = await self.queue.get()
            if event is _DONE:
                break
            yield event

    def cancel(self) -> None:
        """Request cancellation; takes effect at the next engine step boundary."""
        self.request.cancel_requested = True

    @property
    def is_done(self) -> bool:
        return self.request.is_terminal


@dataclass
class _SeqRuntime:
    """Per-sequence runtime data the engine keeps between steps."""

    rng: np.random.Generator
    queue: asyncio.Queue
    pending_logits: np.ndarray | None = None
    state: SequenceState | None = None
    last_token_at: float | None = None


@dataclass
class _FeedRow:
    """A sequence whose sampled token is ready to be fed to the executor."""

    request: GenerationRequest
    runtime: _SeqRuntime
    state: SequenceState
    token_id: int


class FluxEngine:
    """Async engine tying together scheduler, block manager, and executor."""

    def __init__(
        self,
        executor: ModelExecutor,
        tokenize: Callable[[str], list[int]],
        detokenize: Callable[[list[int]], str],
        block_manager: BlockManager,
        limits: SchedulerLimits,
        max_queue_size: int = 128,
        max_prompt_tokens: int = 512,
        max_new_tokens_limit: int | None = None,
        default_seed: int | None = 42,
        metrics: EngineMetrics | None = None,
        enable_batched_execution: bool = True,
    ) -> None:
        self.executor = executor
        self._tokenize = tokenize
        self._detokenize = detokenize
        self.block_manager = block_manager
        self.metrics = metrics or global_metrics
        self.max_prompt_tokens = max_prompt_tokens
        self.max_new_tokens_limit = max_new_tokens_limit
        self.default_seed = default_seed
        self.enable_batched_execution = enable_batched_execution
        self.sampler = Sampler()

        self.scheduler = Scheduler(
            block_manager,
            limits,
            max_queue_size=max_queue_size,
            on_queue_change=self.metrics.set_queue_depth,
        )
        self._runtime: dict[str, _SeqRuntime] = {}
        self._wake = asyncio.Event()
        self._loop_task: asyncio.Task | None = None
        self._stopping = False

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_settings(
        cls, settings: EngineSettings, executor: ModelExecutor | None = None
    ) -> FluxEngine:
        """Build a complete engine from EngineSettings."""
        tokenizer = _make_tokenizer(settings)
        if executor is None:
            executor = _make_executor(settings)
        blocks = BlockManager(
            num_blocks=settings.num_blocks,
            block_size=settings.block_size,
            enable_prefix_cache=settings.enable_prefix_cache,
            # Per-event prefix metrics (same singleton the engine reports to).
            prefix_metrics=global_metrics,
        )
        limits = SchedulerLimits(
            max_batch_size=settings.max_batch_size,
            max_num_batched_tokens=settings.max_num_batched_tokens,
        )
        return cls(
            executor=executor,
            tokenize=tokenizer.encode,
            detokenize=tokenizer.decode,
            block_manager=blocks,
            limits=limits,
            max_queue_size=settings.max_queue_size,
            max_prompt_tokens=settings.max_prompt_tokens,
            max_new_tokens_limit=settings.max_new_tokens,
            default_seed=settings.seed,
            enable_batched_execution=settings.enable_batched_execution,
            metrics=global_metrics,
        )

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self._loop_task is None or self._loop_task.done():
            self._stopping = False
            self._loop_task = asyncio.create_task(self._run_loop(), name="fluxllm-engine")

    async def aclose(self) -> None:
        """Stop the loop and cancel/terminalize any outstanding work."""
        self._stopping = True
        self._wake.set()
        if self._loop_task is not None:
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        for seq in list(self.scheduler.running.values()):
            runtime = self._runtime.get(seq.request.request_id)
            self._finish_request(
                seq.request,
                CompletionReason.CANCELLED,
                runtime.queue if runtime else None,
            )
        for request in list(self.scheduler.queue):
            self._finish_request(request, CompletionReason.CANCELLED, None)
        self._runtime.clear()
        self.metrics.set_queue_depth(0)

    # -- submission -----------------------------------------------------------
    def submit(self, prompt: str, sampling: SamplingParams | None = None) -> RequestHandle:
        """Accept a generation request and return a streaming handle."""
        if sampling is None:
            sampling = SamplingParams()
        if (
            self.max_new_tokens_limit is not None
            and sampling.max_new_tokens > self.max_new_tokens_limit
        ):
            sampling = replace(sampling, max_new_tokens=self.max_new_tokens_limit)
        if sampling.seed is None and self.default_seed is not None:
            sampling = replace(sampling, seed=self.default_seed)
        request = GenerationRequest(prompt=prompt, sampling=sampling)
        request.prompt_token_ids = list(self._tokenize(prompt))
        handle = RequestHandle(engine=self, request=request, queue=asyncio.Queue())

        self.metrics.on_submitted(request.num_prompt_tokens)
        if request.num_prompt_tokens > self.max_prompt_tokens:
            request.fail(f"prompt exceeds {self.max_prompt_tokens} token limit")
            self._push_terminal_events(handle.queue, request)
            return handle
        if len(self.scheduler.queue) >= self.scheduler.max_queue_size:
            self.metrics.on_finished(CompletionReason.ERROR, request.timings)
            raise EngineFullError("engine queue is full")
        if not self.scheduler.submit(request):
            self.metrics.on_finished(CompletionReason.ERROR, request.timings)
            raise EngineFullError("engine queue is full")

        self._runtime[request.request_id] = _SeqRuntime(rng=_rng(sampling.seed), queue=handle.queue)
        self._wake.set()
        return handle

    async def generate(self, prompt: str, sampling: SamplingParams | None = None) -> str:
        """Simplest API: submit and await the full generation."""
        handle = self.submit(prompt, sampling)
        async for _ in handle.stream():
            pass
        return self._detokenize(handle.request.generated_token_ids)

    def stats(self) -> dict:
        """Live engine snapshot for dashboards and health endpoints."""
        return {
            "requests_queued": self.scheduler.queued_count,
            "requests_running": self.scheduler.running_count,
            "scheduler_iterations": self.scheduler.iteration_count,
            "total_admissions": self.scheduler.total_admissions,
            "total_preemptions": self.scheduler.total_preemptions,
            "kv_cache": self.block_manager.snapshot(),
        }

    # -- engine loop ------------------------------------------------------------
    async def _run_loop(self) -> None:
        """Single engine loop: plan → execute steps. Never dies silently."""
        while not self._stopping:
            try:
                progressed = await self._run_iteration()
            except Exception:  # noqa: BLE001 - keep the loop alive and diagnosable
                logger.exception("engine iteration failed")
                progressed = False
            if progressed:
                self._wake.clear()
                continue
            if self.scheduler.queue:
                if not self.scheduler.running:
                    # Nothing runs and the head can never be admitted with
                    # the remaining pool: fail it so the queue can advance.
                    self._fail_stalled_head()
                    continue
                self._wake.clear()
                await self._wake.wait()
                continue
            self._wake.clear()
            await self._wake.wait()

    def _fail_stalled_head(self) -> None:
        head = self.scheduler.queue[0]
        blocks_needed = self.block_manager.blocks_needed(head.total_kv_tokens)
        logger.error(
            "failing stalled request %s: needs %d blocks for %d KV tokens "
            "but the pool has %d free of %d blocks",
            head.request_id,
            blocks_needed,
            head.total_kv_tokens,
            self.block_manager.free_blocks,
            self.block_manager.num_blocks,
        )
        self.scheduler.fail(
            head,
            f"request cannot fit the KV cache: needs {blocks_needed} blocks, "
            f"Pool has {self.block_manager.num_blocks} total blocks",
        )
        runtime = self._runtime.pop(head.request_id, None)
        self.executor.release(head.request_id)
        self._push_terminal_events(runtime.queue if runtime else None, head)
        self.metrics.on_finished(CompletionReason.ERROR, head.timings)

    async def _run_iteration(self) -> bool:
        plan = self.scheduler.plan()
        # Rows whose sampled token is ready to feed to the executor: from
        # prefills (first sampled token) and from decodes (each next token).
        # With batched execution they share ONE forward per iteration.
        feed_rows: list[_FeedRow] = []
        await self._run_prefills(plan.prefills, feed_rows)
        for request in plan.decodes:
            runtime = self._runtime.get(request.request_id)
            if runtime is None or runtime.pending_logits is None:
                continue  # nothing sampled yet (e.g. right after admission)
            if self._check_cancel(request, runtime):
                continue
            if request.num_generated_tokens >= request.sampling.max_new_tokens:
                self._finish_request(request, CompletionReason.LENGTH, runtime.queue)
                continue
            row = self._sample_for_feed(request, runtime, runtime.pending_logits)
            if row is not None:
                feed_rows.append(row)
        if feed_rows:
            await self._execute_feed_rows(feed_rows)
        # Telemetry after committed work: plan() already admitted sequences,
        # so a metrics failure must never strand their prefills.
        self.metrics.on_iteration(self.scheduler.batch_size(), self.block_manager)
        return bool(plan.prefills) or bool(feed_rows)

    # -- batched step execution ----------------------------------------------
    async def _run_prefills(
        self, requests: list[GenerationRequest], feed_rows: list[_FeedRow]
    ) -> None:
        """Prefill planned requests; rows feed forward into this iteration."""
        if not requests:
            return
        ready: list[tuple[GenerationRequest, _SeqRuntime]] = []
        for request in requests:
            runtime = self._runtime.get(request.request_id)
            if runtime is None or self._check_cancel(request, runtime):
                continue
            ready.append((request, runtime))
        if not ready:
            return
        prefill_batch = getattr(self.executor, "prefill_batch", None)
        outs: list | None = None
        if self.enable_batched_execution and len(ready) > 1 and callable(prefill_batch):
            prompts = [
                (request.request_id, request.prompt_token_ids + request.generated_token_ids)
                for request, _ in ready
            ]
            try:
                outs = await asyncio.to_thread(prefill_batch, prompts)
            except Exception:  # noqa: BLE001 - fall back to per-sequence
                logger.exception("batched prefill failed; falling back per sequence")
                outs = None
        if outs is not None:
            for (request, runtime), out in zip(ready, outs, strict=True):
                self._commit_prefill(request, runtime, out, feed_rows)
        else:
            for request, _runtime in ready:
                await self._prefill_step(request)

    def _commit_prefill(
        self,
        request: GenerationRequest,
        runtime: _SeqRuntime,
        out: object,
        feed_rows: list[_FeedRow],
    ) -> None:
        self.scheduler.record_prefill(request, cast("StepOutput", out).state)
        request.transition(RequestState.DECODING)
        runtime.state = cast("StepOutput", out).state
        row = self._sample_for_feed(request, runtime, cast("StepOutput", out).logits)
        if row is not None:
            feed_rows.append(row)

    def _sample_for_feed(
        self, request: GenerationRequest, runtime: _SeqRuntime, logits: np.ndarray
    ) -> _FeedRow | None:
        """Sample the next token, stream it, reserve KV; prepare the feed.

        Mirrors the commitment order of the sequential path: terminal tokens
        finish the request without a forward, and failed KV reservations
        preempt exactly the same way. Streaming and TTFT/inter-token
        decoration happen here, before the executor call, like the
        sequential path.
        """
        token_id = int(self.sampler.sample(logits, request.sampling, runtime.rng))
        if self._terminal_token(request, token_id):
            reason = (
                CompletionReason.STOP_TOKEN
                if self._is_stop_token(token_id, request)
                else CompletionReason.EOS
            )
            request.append_token(token_id)
            self._record_token(request, runtime)
            self._push_token(runtime, token_id)
            self._finish_request(request, reason, runtime.queue)
            return None
        if not self.scheduler.advance(request):
            self._preempt(request)
            return None
        request.append_token(token_id)
        self._record_token(request, runtime)
        self._push_token(runtime, token_id)
        if runtime.state is None:
            # Should be unreachable: prefill always produced executor state.
            self._fail_request(request, RuntimeError("executor state missing before decode"))
            return None
        return _FeedRow(
            request=request,
            runtime=runtime,
            state=runtime.state,
            token_id=token_id,
        )

    async def _execute_feed_rows(self, rows: list[_FeedRow]) -> None:
        decode_batch = getattr(self.executor, "decode_batch", None)
        if self.enable_batched_execution and len(rows) > 1 and callable(decode_batch):
            try:
                outs = await asyncio.to_thread(
                    decode_batch, [(row.state, row.token_id) for row in rows]
                )
            except Exception:  # noqa: BLE001 - batch forward failed before write-back
                logger.exception("batched decode failed; falling back per sequence")
            else:
                for row, out in zip(rows, outs, strict=True):
                    self._commit_decode(row, out)
                self.metrics.on_batched_forward(len(rows))
                return
        for row in rows:
            try:
                out = await asyncio.to_thread(self.executor.decode, row.state, row.token_id)
            except Exception as exc:  # noqa: BLE001
                self._fail_request(row.request, exc)
                continue
            self._commit_decode(row, out)

    def _commit_decode(self, row: _FeedRow, out: object) -> None:
        step = cast("StepOutput", out)
        row.runtime.pending_logits = step.logits
        row.runtime.state = step.state

    # -- step execution ------------------------------------------------------
    async def _prefill_step(self, request: GenerationRequest) -> None:
        runtime = self._runtime[request.request_id]
        if self._check_cancel(request, runtime):
            return
        tokens = request.prompt_token_ids + request.generated_token_ids
        try:
            out = await asyncio.to_thread(self.executor.prefill, request.request_id, tokens)
        except Exception as exc:  # noqa: BLE001 - executor failures fail the request
            self._fail_request(request, exc)
            return
        self.scheduler.record_prefill(request, out.state)
        request.transition(RequestState.DECODING)
        runtime.state = out.state
        await self._sample_and_stream(request, out.logits)

    async def _sample_and_stream(self, request: GenerationRequest, logits: np.ndarray) -> None:
        """Sample the next token, stream it, and feed it back to the executor."""
        runtime = self._runtime[request.request_id]
        token_id = int(self.sampler.sample(logits, request.sampling, runtime.rng))

        if self._terminal_token(request, token_id):
            reason = (
                CompletionReason.STOP_TOKEN
                if self._is_stop_token(token_id, request)
                else CompletionReason.EOS
            )
            request.append_token(token_id)
            self._record_token(request, runtime)
            self._push_token(runtime, token_id)
            self._finish_request(request, reason, runtime.queue)
            return

        # Reserve a KV slot for the token about to be fed.
        if not self.scheduler.advance(request):
            self._preempt(request)
            return
        request.append_token(token_id)
        self._record_token(request, runtime)
        self._push_token(runtime, token_id)
        if runtime.state is None:
            # Should be unreachable: prefill always produced executor state.
            self._fail_request(request, RuntimeError("executor state missing before decode"))
            return
        try:
            out = await asyncio.to_thread(self.executor.decode, runtime.state, token_id)
        except Exception as exc:  # noqa: BLE001
            self._fail_request(request, exc)
            return
        runtime.pending_logits = out.logits
        runtime.state = out.state

    # -- helpers: state transitions and teardown ------------------------------
    def _terminal_token(self, request: GenerationRequest, token_id: int) -> bool:
        if token_id == self.executor.eos_token_id:
            return True
        return token_id in request.sampling.stop_token_ids

    @staticmethod
    def _is_stop_token(token_id: int, request: GenerationRequest) -> bool:
        return token_id in request.sampling.stop_token_ids

    def _check_cancel(self, request: GenerationRequest, runtime: _SeqRuntime) -> bool:
        if request.cancel_requested and not request.is_terminal:
            self._finish_request(request, CompletionReason.CANCELLED, runtime.queue)
            return True
        return request.is_terminal

    def _preempt(self, request: GenerationRequest) -> None:
        logger.info("preempting %s: KV pool exhausted", request.request_id)
        runtime = self._runtime.get(request.request_id)
        if runtime is not None:
            runtime.pending_logits = None
            runtime.state = None
        self.executor.release(request.request_id)
        self.scheduler.preempt(request)
        self.metrics.inc_preemptions()

    def _finish_request(
        self,
        request: GenerationRequest,
        reason: CompletionReason,
        queue: asyncio.Queue | None = None,
    ) -> None:
        """Terminalize a request: lifecycle, resources, stream events, metrics."""
        runtime = self._runtime.pop(request.request_id, None)
        if not request.is_terminal:
            self.scheduler.complete(request, reason)
        self.executor.release(request.request_id)
        self.metrics.on_finished(request.completion_reason or reason, request.timings)
        target_queue = queue if queue is not None else (runtime.queue if runtime else None)
        self._push_terminal_events(target_queue, request)

    def _fail_request(self, request: GenerationRequest, exc: Exception) -> None:
        logger.exception("request %s failed: %s", request.request_id, exc)
        runtime = self._runtime.pop(request.request_id, None)
        self.scheduler.fail(request, str(exc)[:280])
        self.executor.release(request.request_id)
        self.metrics.on_finished(CompletionReason.ERROR, request.timings)
        self._push_terminal_events(runtime.queue if runtime else None, request)

    # -- events / metrics -------------------------------------------------------
    def _record_token(self, request: GenerationRequest, runtime: _SeqRuntime) -> None:
        now = time.perf_counter()
        if runtime.last_token_at is None:
            request.mark_first_token()
            self.metrics.record_ttft(request.timings)
        else:
            self.metrics.record_inter_token(now - runtime.last_token_at)
        runtime.last_token_at = now
        self.metrics.on_token(request.num_generated_tokens)

    @staticmethod
    def _push_token(runtime: _SeqRuntime, token_id: int) -> None:
        runtime.queue.put_nowait(StreamEvent(type="token", token_id=token_id))

    def _push_terminal_events(
        self, queue: asyncio.Queue | None, request: GenerationRequest
    ) -> None:
        if queue is None:
            return
        timings = request.timings
        meta = {
            "ttft_seconds": timings.ttft,
            "prefill_seconds": timings.prefill_latency,
            "decode_seconds": timings.decode_latency,
            "total_seconds": timings.total_latency,
            "queue_delay_seconds": timings.queue_delay,
            "prompt_tokens": request.num_prompt_tokens,
            "generated_tokens": request.num_generated_tokens,
        }
        queue.put_nowait(
            StreamEvent(
                type="status",
                state=request.state.value,
                reason=request.completion_reason.value if request.completion_reason else None,
                meta=meta,
            )
        )
        queue.put_nowait(_DONE)


# -- module-level wiring helpers ------------------------------------------------


def _rng(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**32 - 1))
    return np.random.default_rng(seed)


class _DetTokenizer:
    """Tokenizer adapter for deterministic simulation mode."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        from fluxllm.engine.tokens import simple_tokenize

        return simple_tokenize(text, self.vocab_size)

    def decode(self, token_ids: list[int]) -> str:
        from fluxllm.engine.tokens import simple_detokenize

        return simple_detokenize(token_ids)


class _HFTokenizerAdapter:
    def __init__(self, model_name: str, trust_remote_code: bool = False) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=True))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))


def _make_tokenizer(
    settings: EngineSettings,
) -> _DetTokenizer | _HFTokenizerAdapter:
    if settings.deterministic:
        return _DetTokenizer(settings.vocab_size)
    return _HFTokenizerAdapter(settings.model_name, settings.trust_remote_code)


def _make_executor(settings: EngineSettings) -> ModelExecutor:
    if settings.deterministic:
        from fluxllm.executor.deterministic import DeterministicExecutor

        return DeterministicExecutor(
            vocab_size=settings.vocab_size,
            eos_token_id=getattr(settings, "demo_eos_token_id", None),
        )
    from fluxllm.executor.hf_executor import HFModelExecutor

    return HFModelExecutor(
        model_name=settings.model_name,
        dtype=settings.dtype,
        device=settings.device,
        trust_remote_code=settings.trust_remote_code,
    )


__all__ = [
    "EngineFullError",
    "FluxEngine",
    "RequestHandle",
    "StreamEvent",
]
