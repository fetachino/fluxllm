"""Render benchmark artifacts into PNG charts for the documentation.

Reads the newest ``benchmarks/results/run_*.json`` and writes:

- ``docs/images/benchmark_throughput.png`` — throughput by concurrency with
  the sequential transformers baseline as a reference line;
- ``docs/images/benchmark_ttft.png`` — TTFT p50/p95 by concurrency.

Usage: python scripts/make_charts.py
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless: no display server required
import matplotlib.pyplot as plt  # noqa: E402

RESULTS_GLOB = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results", "run_*.json")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")


def newest_run() -> dict | None:
    matches = sorted(glob.glob(RESULTS_GLOB))
    if not matches:
        return None
    with open(matches[-1], encoding="utf-8") as handle:
        return json.load(handle)


def make_charts(payload: dict) -> None:
    runs = payload["runs"]
    flux = [r for r in runs if r["mode"] == "fluxllm"]
    seq = [r for r in runs if r["mode"] == "sequential_transformers"]
    batched = {
        r["concurrency"]: r
        for r in flux
        if r.get("batched_execution", True) is not False
    }
    per_sequence = {
        r["concurrency"]: r
        for r in flux
        if r.get("batched_execution") is False
    }
    levels = sorted({r["concurrency"] for r in flux})
    os.makedirs(OUT_DIR, exist_ok=True)

    env = payload["env"]
    caption = (
        f"CPU cores={env['cpu_count']}, torch threads={env['torch_threads']}, "
        f"{env['dtype']} on {env['device']}"
    )

    # -- throughput ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if per_sequence:
        width = 0.38
        xs = list(range(len(levels)))
        ax.bar(
            [x - width / 2 for x in xs],
            [batched[c]["throughput_tokens_per_s"] for c in levels],
            width,
            color="#2563eb",
            label="FluxLLM batched decode",
        )
        ax.bar(
            [x + width / 2 for x in xs],
            [per_sequence[c]["throughput_tokens_per_s"] for c in levels],
            width,
            color="#93c5fd",
            label="FluxLLM per-sequence decode",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(c) for c in levels])
    else:
        ax.bar(
            [str(c) for c in levels],
            [batched[c]["throughput_tokens_per_s"] for c in levels],
            color="#2563eb",
            label="FluxLLM (continuous batching)",
        )
    if seq:
        baseline = seq[0]["throughput_tokens_per_s"]
        ax.axhline(baseline, color="#9ca3af", linestyle="--", linewidth=1.2,
                   label=f"sequential transformers baseline ({baseline:.2f} tok/s)")
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("throughput (tokens/s)")
    ax.set_title("FluxLLM throughput vs concurrency\n" + caption, fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "benchmark_throughput.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path} from {len(runs)} runs")

    # -- TTFT ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.35
    xs = list(range(len(levels)))
    if per_sequence:
        ax.bar(
            [x - width / 2 for x in xs],
            [batched[c]["ttft_p50_seconds"] for c in levels],
            width,
            color="#2563eb",
            label="batched TTFT p50",
        )
        ax.bar(
            [x + width / 2 for x in xs],
            [per_sequence[c]["ttft_p50_seconds"] for c in levels],
            width,
            color="#93c5fd",
            label="per-sequence TTFT p50",
        )
    else:
        ax.bar(
            [x - width / 2 for x in xs],
            [batched[c]["ttft_p50_seconds"] for c in levels],
            width,
            color="#2563eb",
            label="TTFT p50",
        )
        ax.bar(
            [x + width / 2 for x in xs],
            [batched[c]["ttft_p95_seconds"] for c in levels],
            width,
            color="#93c5fd",
            label="TTFT p95",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in levels])
    ax.set_xlabel("concurrent requests")
    ax.set_ylabel("time to first token (s)")
    ax.set_title("TTFT by concurrency\n" + caption, fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "benchmark_ttft.png")
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> None:
    payload = newest_run()
    if payload is None:
        raise SystemExit("no benchmark artifacts found in benchmarks/results/")
    make_charts(payload)


if __name__ == "__main__":
    main()
