"""Engine-level settings loaded from environment variables.

All runtime knobs live here so the engine, API, and benchmarks can be
configured without code changes (12-factor style). No secrets are defined:
FluxLLM holds no credentials by design.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Dtype = Literal["float32", "float16", "bfloat16"]


class EngineSettings(BaseSettings):
    """Configuration for building a FluxLLM engine."""

    model_config = SettingsConfigDict(env_prefix="FLUXLLM_", env_file=".env", extra="ignore")

    # Model execution -------------------------------------------------------
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: Dtype = "float32"
    device: str = "auto"
    trust_remote_code: bool = False
    deterministic: bool = False  # deterministic test executor instead of a real model
    vocab_size: int = 64  # only used by the deterministic executor
    seed: int = 42

    # Scheduler -------------------------------------------------------------
    max_batch_size: int = 8
    max_num_batched_tokens: int = 512  # prefill token budget per scheduler iteration
    max_queue_size: int = 128
    scheduler_policy: Literal["fifo", "longest_queue_first"] = "fifo"
    enable_batched_execution: bool = True  # one forward per phase across sequences

    # KV cache --------------------------------------------------------------
    block_size: int = 16  # tokens per logical block
    num_blocks: int = 64  # logical KV blocks available
    enable_prefix_cache: bool = True

    # Generation limits (abuse protection) ----------------------------------
    max_prompt_tokens: int = 512
    max_new_tokens: int = 128

    # Observability ---------------------------------------------------------
    log_level: str = "INFO"
    log_prompts: bool = False  # never log user prompts unless explicitly enabled

    # API -------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
