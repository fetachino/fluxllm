"""Throughput benchmark: HF sequential generate vs FluxLLM continuous batching.

Methodology (full details in BENCHMARKS.md):

- one fixed prompt set, greedy decoding (temperature 0), identical
  ``max_new_tokens`` for every mode;
- baseline = ``transformers`` ``model.generate`` run one request at a time
  (the standard naive way to serve fixed prompts);
- FluxLLM = the continuous-batching engine with ``c`` concurrent
  submissions (slots free and immediately re-fill, no batch drain);
- results are appended to ``benchmarks/results/<timestamp>.json`` and a
  CSV summary; nothing in this script fabricates numbers — anything not
  measured reads as null.

Example:

    python benchmarks/benchmark_throughput.py \
        --concurrency 1 2 4 8 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")

# Import the in-repo fluxllm package regardless of install state.
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_PROMPTS = [
    "Explain what continuous batching is in three sentences.",
    "Write a short Python function to reverse a string.",
    "Summarize the main idea of consistent hashing.",
    "What are the tradeoffs of float16 versus bfloat16?",
    "Draft a polite deadline-extension email.",
    "Name three benefits of unit testing.",
    "Describe how HTTP streaming works.",
    "Give an example of a good cache key.",
]


@dataclass
class RequestResult:
    request_id: str
    total_seconds: float
    tokens: int
    tokens_per_second: float
    ttft_seconds: float | None = None
    reason: str | None = None


@dataclass
class RunResult:
    mode: str  # "sequential_transformers" | "fluxllm"
    concurrency: int
    wall_seconds: float
    total_tokens: int
    throughput_tokens_per_s: float
    arrivals: int
    preemptions: int | None = None
    ttft_p50_seconds: float | None = None
    ttft_p95_seconds: float | None = None
    e2e_p50_seconds: float | None = None
    e2e_p95_seconds: float | None = None
    per_request: list[RequestResult] = field(default_factory=list)


def _git_commit_sha() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:  # noqa: BLE001 - sha is informational only
        return None


def env_snapshot() -> dict:
    import torch
    import transformers

    return {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_name": MODEL_NAME,
        "dtype": "float32",
        "device": "cpu",
        "commit_sha": _git_commit_sha(),
    }


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


# -- sequential transformers baseline ---------------------------------------
def run_sequential_transformers(
    prompts: list[str], max_new_tokens: int, warmup: int = 0
) -> RunResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for prompt in prompts[:warmup]:
        inputs = tokenizer(prompt, return_tensors="pt")
        model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    per_request: list[RequestResult] = []
    wall_start = time.perf_counter()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed = time.perf_counter() - t0
        tokens = int(output.shape[1]) - prompt_len
        per_request.append(
            RequestResult(
                request_id="sequential",
                total_seconds=elapsed,
                tokens=tokens,
                tokens_per_second=tokens / elapsed,
            )
        )
    wall = time.perf_counter() - wall_start
    total_tokens = sum(r.tokens for r in per_request)
    e2e = [r.total_seconds for r in per_request]
    return RunResult(
        mode="sequential_transformers",
        concurrency=1,
        wall_seconds=wall,
        total_tokens=total_tokens,
        throughput_tokens_per_s=total_tokens / wall,
        arrivals=len(prompts),
        e2e_p50_seconds=_percentile(e2e, 50),
        e2e_p95_seconds=_percentile(e2e, 95),
        per_request=per_request,
    )


# -- FluxLLM continuous batching ---------------------------------------------
def run_fluxllm(
    prompts: list[str],
    max_new_tokens: int,
    concurrency: int,
    warmup: int = 0,
) -> RunResult:
    from fluxllm.config.settings import EngineSettings
    from fluxllm.engine.engine import FluxEngine
    from fluxllm.executor.hf_executor import HFModelExecutor
    from fluxllm.sampling.config import SamplingParams

    settings = EngineSettings(
        model_name=MODEL_NAME,
        dtype="float32",
        device="cpu",
        enable_prefix_cache=True,
        block_size=16,
        num_blocks=96,
        max_batch_size=concurrency,
        max_num_batched_tokens=4096,
        max_queue_size=64,
        max_prompt_tokens=512,
        max_new_tokens=max_new_tokens,
        seed=42,
    )
    executor = HFModelExecutor(
        model_name=MODEL_NAME, dtype="float32", device="cpu"
    )
    engine = FluxEngine.from_settings(settings, executor=executor)

    async def one(prompt: str) -> RequestResult:
        handle = engine.submit(
            prompt,
            SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0, top_k=1),
        )
        async for _event in handle.stream():  # stream to completion
            pass
        timings = handle.request.timings
        tokens = handle.request.num_generated_tokens
        total = timings.total_latency or 0.0
        return RequestResult(
            request_id=handle.request.request_id,
            ttft_seconds=timings.ttft,
            total_seconds=total,
            tokens=tokens,
            tokens_per_second=tokens / total if total > 0 else float("nan"),
            reason=handle.request.completion_reason.value
            if handle.request.completion_reason
            else None,
        )

    async def main() -> tuple[list[RequestResult], float]:
        semaphore_slots = asyncio.Semaphore(concurrency)

        async def bounded(prompt: str) -> RequestResult:
            async with semaphore_slots:
                return await one(prompt)

        engine.start()
        try:
            for prompt in prompts[:warmup]:
                await one(prompt)
            start = time.perf_counter()
            results = await asyncio.gather(
                *(bounded(prompt) for prompt in prompts)
            )
            elapsed = time.perf_counter() - start
        finally:
            await engine.aclose()
        return results, elapsed

    per_request, wall = asyncio.run(main())
    preemptions = engine.scheduler.total_preemptions
    tokens_total = sum(r.tokens for r in per_request)
    ttfts = [r.ttft_seconds for r in per_request if r.ttft_seconds is not None]
    e2e = [r.total_seconds for r in per_request]
    result = RunResult(
        mode="fluxllm",
        concurrency=concurrency,
        wall_seconds=wall,
        total_tokens=tokens_total,
        throughput_tokens_per_s=tokens_total / wall,
        arrivals=len(prompts),
        preemptions=preemptions,
        ttft_p50_seconds=_percentile(ttfts, 50) if ttfts else None,
        ttft_p95_seconds=_percentile(ttfts, 95) if ttfts else None,
        e2e_p50_seconds=_percentile(e2e, 50),
        e2e_p95_seconds=_percentile(e2e, 95),
        per_request=per_request,
    )
    return result


def write_csv(runs: list[RunResult], path: str) -> None:
    headers = [
        "mode",
        "concurrency",
        "wall_seconds",
        "total_tokens",
        "throughput_tokens_per_s",
        "arrivals",
        "preemptions",
        "ttft_p50_seconds",
        "ttft_p95_seconds",
        "e2e_p50_seconds",
        "e2e_p95_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for run in runs:
            row = {h: getattr(run, h) for h in headers}
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="FluxLLM concurrency levels to measure",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--skip-sequential",
        action="store_true",
        help="skip the sequential baseline run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny wiring-validation run (2 prompts, 4 tokens)",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    max_new_tokens = args.max_new_tokens
    if args.quick:
        prompts = DEFAULT_PROMPTS[:2]
        max_new_tokens = 4
        args.warmup = 0
        args.skip_sequential = True
        args.concurrency = [2]

    print(f"env: {env_snapshot()['cpu_count']} cpus, "
          f"threads={env_snapshot()['torch_threads']}, "
          f"model={MODEL_NAME} (float32, cpu)")

    runs: list[RunResult] = []
    if not args.skip_sequential:
        print("running sequential transformers baseline ...")
        runs.append(run_sequential_transformers(prompts, max_new_tokens, warmup=args.warmup))
        last = runs[-1]
        print(f"  -> {last.total_tokens} tokens in {last.wall_seconds:.1f}s "
              f"({last.throughput_tokens_per_s:.2f} tok/s)")

    for level in args.concurrency:
        print(f"running FluxLLM concurrency={level} ...")
        runs.append(
            run_fluxllm(prompts, max_new_tokens, level, warmup=args.warmup)
        )
        last = runs[-1]
        preemptions = last.preemptions
        print(
            f"  -> {last.total_tokens} tokens in {last.wall_seconds:.1f}s "
            f"({last.throughput_tokens_per_s:.2f} tok/s, "
            f"ttft_p50={last.ttft_p50_seconds:.2f}s, preemptions={preemptions})"
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(RESULTS_DIR, f"run_{stamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"summary_{stamp}.csv")
    payload = {"env": env_snapshot(), "runs": [asdict(run) for run in runs]}
    with open(json_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)
    write_csv(runs, csv_path)
    print(f"artifacts: {json_path}, {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
