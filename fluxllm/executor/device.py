"""Device and dtype helpers (PyTorch optional)."""

from __future__ import annotations

from typing import Any


def detect_device(preferred: str = "auto") -> str:
    """Resolve ``auto`` to CUDA (if available) and otherwise CPU.

    Import of torch is deferred so the deterministic executor and scheduler
    tests work in environments without torch installed.
    """
    if preferred != "auto":
        return preferred
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def resolve_dtype(name: str, device: str) -> Any:
    """Map a dtype name string to the torch dtype for ``device``."""
    import torch

    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype: {name}")
    return table[name]
