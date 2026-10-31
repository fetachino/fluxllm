"""FluxLLM: a high-throughput LLM inference engine.

Implements continuous request scheduling, block-based KV-cache management,
prefill/decode separation, sampling, and streaming generation on top of
PyTorch and Hugging Face model weights.
"""

__version__ = "0.1.0"
