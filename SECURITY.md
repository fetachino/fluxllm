# Security

## Threat model

FluxLLM runs as an HTTP service that accepts arbitrary text prompts and
generates completions. Typical deployment is inside a trusted boundary
(single-tenant container Coolify/Docker/VM) reachable from the owner's
network — but the code is hardened against the realistic failure cases of
exposed inference endpoints.

## What the server itself does by design

- **No secrets.** FluxLLM stores no credentials; settings come from
  environment variables (`FLUXLLM_*`, 12-factor). Hugging Face model
  downloads are unauthenticated (public models); `HF_TOKEN` is never read.
- **No prompt logging** unless explicitly enabled (`FLUXLLM_LOG_PROMPTS=0`
  by default). Metrics never carry prompt or output content.
- **Container hardening** (in `compose.yaml`): read-only root filesystem,
  writable `/tmp` via tmpfs, non-root user (`1001:1001`), no privileged
  mode, model cache on a named volume (`hf-cache` mounted at `/models-cache`).
- **Non-root image**: the Dockerfile drops to an unmanaged system user; the
  healthcheck curls localhost only.

## Abuse limits (implemented)

- `FLUXLLM_MAX_PROMPT_TOKENS` (default 512) — oversized prompts are
  rejected fast with a clear error, not queued.
- `FLUXLLM_MAX_NEW_TOKENS` (default 128) — clamps per-request generation
  regardless of client `max_tokens`.
- `FLUXLLM_MAX_QUEUE_SIZE` (default 128) — queue-full rejection returns
  HTTP 503 (client backpressure, not unbounded memory).
- `FLUXLLM_NUM_BLOCKS` — the KV pool is a hard capacity ceiling; the
  scheduler never admits beyond it, and pool exhaustion preempts rather
  than overcommits.

## Denial-of-service surface and mitigations

- **Long generations** are the main DoS vector for any LLM server. The
  engine's hard per-request token clamp, batch-size cap, and KV token
  budget bound the worst-case work a single request can cause.
- **Concurrent connection flood** — the queue-size cap and 503s bound
  scheduler memory; cancellation paths release blocks promptly (verified
  by tests that assert a fully free pool after cancel/fail).
- **Prompt-size parsing** — prompts are validated against
  `max_prompt_tokens` after tokenization and rejected before allocation.

## Known gaps (documented honestly)

- **No authentication layer** — the API is unauthenticated by design for
  local/lab use. For any shared deployment: put an authenticating reverse
  proxy (basic auth, OIDC proxy) in front; Coolify front-ends can do this.
- **No rate limiting per client** — rely on the reverse proxy.
- **No TLS** — the server speaks plain HTTP; terminate TLS at the proxy.
- **`trust_remote_code` defaults to false**; enabling it (`FLUXLLM_` env)
  would execute model-repo code — do not enable it for untrusted repos.
- **Model supply chain**: `AutoModelForCausalLM.from_pretrained` trusts the
  Hub for the configured model; pin your model and revision in a private
  deployment and use your own scanned cache (`HF_HOME`).

## Reporting

Open a GitHub issue for anything you find; for anything sensitive, use
GitHub's private security advisory feature on this repo.
