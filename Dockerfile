# syntax=docker/dockerfile:1
# FluxLLM API image. CPU-only torch keeps the image broadly deployable
# (Coolify, plain Docker, CI); larger GPU models run on CUDA hosts natively.
FROM python:3.11-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models-cache

WORKDIR /app

RUN groupadd --system fluxllm && useradd --system --gid fluxllm fluxllm \
    && mkdir -p /models-cache /app && chown -R fluxllm:fluxllm /models-cache /app

# CPU-only PyTorch wheel index keeps the image ~1.5GB instead of pulling CUDA.
# torch is installed first from the CPU index; the torch extra then pulls its
# remaining members (transformers, accelerate, safetensors) without re-installing torch.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY fluxllm ./fluxllm
COPY demo ./demo
RUN pip install --no-cache-dir -e ".[api,torch]"

USER fluxllm

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=4)" || exit 1

CMD ["uvicorn", "fluxllm.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# -----
# Benchmark image: same runtime plus the benchmark harness and matplotlib.
FROM api AS benchmark

USER root
COPY benchmarks ./benchmarks
COPY scripts ./scripts
RUN pip install --no-cache-dir matplotlib && chown -R fluxllm:fluxllm /app
USER fluxllm

CMD ["python", "benchmarks/benchmark_throughput.py", "--help"]
