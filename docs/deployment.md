# Deployment

FluxLLM ships a production-style container story: multi-stage Dockerfile,
hardened compose file, and a Coolify path. All of these were built and
exercised against a real local Docker daemon (the API image was built and
smoke-tested serving `/health/ready` and `/demo`); the Coolify steps are
written for the exact image this repo produces.

## Image layout

| target | contents | use |
| --- | --- | --- |
| `api` | CPU PyTorch, engine + API + demo page, non-root user, healthcheck | the serving image |
| `benchmark` | api runtime + benchmark harness + matplotlib | perf runs in-container |

```bash
docker build --target api -t fluxllm-api .
docker build --target benchmark -t fluxllm-bench .
```

The base image pins `python:3.11-slim` and installs the CPU-only torch
wheel index (`download.pytorch.org/whl/cpu`), keeping the image ~1.5 GB
instead of pulling CUDA runtime. On a CUDA host, build with the regular
torch wheel and pass `FLUXLLM_DEVICE=cuda`.

## Run standalone

```bash
docker run --rm -p 8000:8000 --name fluxllm fluxllm-api
```

First start downloads `Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) into the image's
`HF_HOME` (`/models-cache`) — mount a volume there so weights survive
restarts:

```bash
docker volume create fluxllm-hf-cache
docker run --rm -p 8000:8000 \
  -v fluxllm-hf-cache:/models-cache \
  --name fluxllm fluxllm-api
```

Endpoints: `/health/live`, `/health/ready`, `/demo/`, `/v1/completions`,
`/v1/chat/completions`, `/stats`, `/metrics`.

## Run with compose (hardened)

```bash
docker compose up     # from the repo root
```

`compose.yaml` applies the container hardening from SECURITY.md:

- non-root user `1001:1001`, read-only root filesystem (`read_only: true`);
- writable `/tmp` (home) via tmpfs, because transformers needs a real
  writable HOME;
- model cache on the named volume `hf-cache` mounted at `/models-cache`;
- no privileged mode and a healthcheck hitting `/health/live`.

Validate the file: `docker compose config --quiet` (exit code 0 = valid).

## Coolify

Coolify deploys the same image; there is nothing FluxLLM-specific to
install. Steps that were checked against Coolify's semantics for this repo:

1. **New Resource → Docker Compose (blank)** and paste `compose.yaml`, or
   **Docker Compose from git repo** pointing at this repository with
   `compose.yaml` selected.
2. Alternatively **New Resource → Docker Image** with image
   `ghcr.io/fetachino/fluxllm-api` (see "Publishing the image" below), port
   8000, and Coolify's persistent volume for `/models-cache`.
3. Environment variables (all optional; defaults are sensible): see
   settings in `fluxllm/config/settings.py`, e.g.
   - `FLUXLLM_DETERMINISTIC=1` for the labeled simulation mode (good for
     smoke tests),
   - `FLUXLLM_MODEL_NAME`, `FLUXLLM_DTYPE` (float32 on CPU; bfloat16 on
     GPUs), `FLUXLLM_DEVICE=auto|cpu|cuda`,
   - capacity knobs: `FLUXLLM_MAX_BATCH_SIZE`, `FLUXLLM_NUM_BLOCKS`,
     `FLUXLLM_BLOCK_SIZE`, `FLUXLLM_MAX_QUEUE_SIZE`.
4. **Health check**: point it at `/health/live` (cheap, no engine wait) —
   the compose file already wires this; have the start-period high enough
   for the first model download (~2–3 min for 1 GB).
5. Resource sizing: CPU-only serving of the 0.5B float32 model measured
   ~6.5 tok/s with one 16-thread CPU core group busy — size cores for your
   target concurrency of *iterations*, not requests (throughput does not
   scale with requests; see BENCHMARKS.md).
6. Domain + TLS are handled by Coolify's Traefik integration; the API
   needs no special headers. Because responses are long-lived SSE streams,
   give the proxy a generous idle/stream timeout (increased proxy idle
   timeouts are the usual first fix if streams are cut).

Note: this repository has no published Coolify-managed deployment yet.
When one exists, its URL and configuration will be listed here — nothing
is claimed before it is running.

## Publishing the image

The repo currently builds the image in CI (no push yet). To publish:

1. Create a GitHub Token with `write:packages` and add it as repo secret
   `GITHUB_TOKEN` access (default in Actions) — then add a `docker` job
   step:
   ```yaml
   - uses: docker/login-action@v3
     with:
       registry: ghcr.io
       username: ${{ github.actor }}
       password: ${{ secrets.GITHUB_TOKEN }}
   - uses: docker/build-push-action@v6
     with:
       context: .
       target: api
       push: true
       tags: ghcr.io/fetachino/fluxllm-api:latest
   ```
2. Tag releases as `v*` so the version labels in the image are meaningful.

## Grafana / monitoring

`/metrics` is a standard Prometheus text endpoint. Useful panels to start:
`fluxllm_requests_queued`, `fluxllm_requests_running`,
`fluxllm_scheduler_batch_size`, `fluxllm_kv_cache_utilization_ratio`,
`fluxllm_preemptions_total` (alert if it grows),
`fluxllm_prefix_cache_hit_rate`, and the TTFT/e2e histograms.
