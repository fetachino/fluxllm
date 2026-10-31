/* FluxLLM demo client.
 *
 * Talks only to the public HTTP API:
 * - POST /v1/chat/completions (stream: true) — OpenAI-style SSE
 * - GET /stats, GET /metrics, GET /health/ready
 * Aborting the fetch (Stop button) disconnects the stream; the API layer
 * cancels the engine-side request on client disconnect.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const el = {
  health: $("health"),
  prompt: $("prompt"),
  maxTokens: $("maxTokens"),
  temp: $("temp"),
  seed: $("seed"),
  send: $("send"),
  stop: $("stop"),
  output: $("output"),
  chipTtft: $("chipTtft"),
  chipChunks: $("chipChunks"),
  chipRate: $("chipRate"),
  chipFinish: $("chipFinish"),
  queued: $("queued"),
  running: $("running"),
  batch: $("batch"),
  iterations: $("iterations"),
  admitted: $("admitted"),
  preempted: $("preempted"),
  blocksUsed: $("blocksUsed"),
  kvBar: $("kvBar"),
  kvUtil: $("kvUtil"),
  genTokens: $("genTokens"),
  prefixRate: $("prefixRate"),
  lanes: $("lanes"),
  burst: $("burst"),
};

/* ---------- engine polling ---------- */

function gaugeValue(metricsText, name) {
  const re = new RegExp("^" + name + "\\s+([-0-9.eE+]+)\\s*$", "m");
  const match = metricsText.match(re);
  return match ? parseFloat(match[1]) : null;
}

async function refreshEngine() {
  try {
    const [stats, metricsText] = await Promise.all([
      fetch("/stats").then((r) => r.json()),
      fetch("/metrics").then((r) => r.text()),
    ]);

    el.queued.textContent = stats.requests_queued ?? "—";
    el.running.textContent = stats.requests_running ?? "—";
    el.iterations.textContent = stats.scheduler_iterations ?? "—";
    el.admitted.textContent = stats.total_admissions ?? "—";
    el.preempted.textContent = stats.total_preemptions ?? "—";

    const kv = stats.kv_cache || {};
    el.blocksUsed.textContent =
      kv.used_blocks !== undefined ? `${kv.used_blocks} / ${kv.num_blocks}` : "—";
    if (typeof kv.utilization === "number") {
      el.kvBar.style.width = `${(kv.utilization * 100).toFixed(1)}%`;
      el.kvUtil.textContent = `${(kv.utilization * 100).toFixed(1)}%`;
    }
    const gen = gaugeValue(metricsText, "fluxllm_generated_tokens_total");
    if (gen !== null) el.genTokens.textContent = Math.round(gen);
    const rate = gaugeValue(metricsText, "fluxllm_prefix_cache_hit_rate");
    if (rate !== null) el.prefixRate.textContent = `${(rate * 100).toFixed(0)}%`;
    const batch = gaugeValue(metricsText, "fluxllm_scheduler_batch_size");
    if (batch !== null) el.batch.textContent = Math.round(batch);
  } catch {
    /* server restarting; keep last values */
  }
}

setInterval(refreshEngine, 1000);
refreshEngine();

async function refreshHealth() {
  try {
    const info = await fetch("/health/ready").then((r) => r.json());
    if (info.engine_running) {
      el.health.textContent = `engine ok · ${info.model ?? "?"} · ${info.device ?? "cpu"}`;
      el.health.className = "chip good";
    } else {
      el.health.textContent = "engine degraded";
      el.health.className = "chip bad";
    }
  } catch {
    el.health.textContent = "api unreachable";
    el.health.className = "chip bad";
  }
}
refreshHealth();
setInterval(refreshHealth, 5000);

/* ---------- OpenAI-style SSE consumption ---------- */

async function streamChat(body, onTokenPiece, onFirstToken) {
  const started = performance.now();
  let gotFirst = false;

  const controller = new AbortController();
  streamChat.currentController = controller;

  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: body.prompt }],
      stream: true,
      max_tokens: body.maxTokens,
      temperature: body.temperature,
      seed: body.seed ?? undefined,
    }),
    signal: controller.signal,
  });

  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let content = "";
  let finish = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") {
        return { content, finish, elapsed: performance.now() - started };
      }
      const chunk = JSON.parse(payload);
      const piece = chunk.choices?.[0]?.delta?.content || "";
      finish = chunk.choices?.[0]?.finish_reason || finish;
      if (piece) {
        if (!gotFirst) {
          gotFirst = true;
          onFirstToken?.(performance.now() - started);
        }
        content += piece;
        onTokenPiece?.(content, performance.now() - started);
      }
    }
  }
  return { content, finish, elapsed: performance.now() - started };
}

function setRequestDone(result, pieces) {
  const seconds = result.elapsed / 1000;
  $("chipTtft").textContent = streamChat._lastTtft
    ? `TTFT ${(streamChat._lastTtft / 1000).toFixed(2)}s`
    : "TTFT —";
  $("chipChunks").textContent = `pieces ${pieces}`;
  $("chipRate").textContent = `rate ${(pieces / Math.max(seconds, 0.001)).toFixed(1)} tok/s`;
  $("chipFinish").textContent = `finish ${result.finish ?? "—"}`;
  $("send").disabled = false;
  $("stop").disabled = true;
}

/* ---------- single request ---------- */

el.send.addEventListener("click", async () => {
  const prompt = el.prompt.value.trim();
  if (!prompt) return;

  el.send.disabled = true;
  el.stop.disabled = false;
  el.output.textContent = "";
  $("chipTtft").textContent = "TTFT —";
  $("chipChunks").textContent = "pieces —";
  $("chipRate").textContent = "rate —";
  $("chipFinish").textContent = "finish —";
  streamChat._lastTtft = null;

  let pieces = 0;
  try {
    const result = await streamChat(
      {
        prompt,
        maxTokens: Number(el.maxTokens.value) || 24,
        temperature: Number(el.temp.value) || 0.2,
        seed: el.seed.value === "" ? undefined : Number(el.seed.value),
      },
      (content) => {
        pieces += 1;
        el.output.textContent = content;
      },
      (ttft) => {
        streamChat._lastTtft = ttft;
        $("chipTtft").textContent = `TTFT ${(ttft / 1000).toFixed(2)}s`;
      }
    );
    setRequestDone(result, pieces);
  } catch (err) {
    if (err.name === "AbortError") {
      $("chipFinish").textContent = "finish cancelled";
    } else {
      el.output.textContent += `\n[request failed: ${err.message}]`;
      $("chipFinish").textContent = "finish error";
    }
    $("send").disabled = false;
    $("stop").disabled = true;
  }
});

el.stop.addEventListener("click", () => {
  streamChat.currentController?.abort();
});

/* ---------- burst mode ---------- */

const BURST_PROMPTS = [
  "Name one benefit of unit tests.",
  "Give a one-sentence definition of caching.",
  "Describe HTTP streaming in one sentence.",
  "What does idempotent mean? One sentence.",
];

function renderLanes() {
  el.lanes.innerHTML = "";
  BURST_PROMPTS.forEach((_, i) => {
    const lane = document.createElement("div");
    lane.className = "lane";
    lane.id = `lane-${i}`;
    lane.innerHTML = `
      <div class="num">#${i + 1}</div>
      <div class="out">—</div>
      <div class="meta">waiting</div>
    `;
    el.lanes.appendChild(lane);
  });
}

renderLanes();

const burstState = { running: false };

async function runBurst() {
  if (burstState.running) return;
  burstState.running = true;
  renderLanes();
  el.burst.disabled = true;

  const started = performance.now();

  const runs = BURST_PROMPTS.map(async (prompt, i) => {
    const lane = $(`lane-${i}`);
    let pieces = 0;
    let ttft = null;
    const laneStarted = performance.now();
    try {
      const result = await streamChat(
        { prompt, maxTokens: 16, temperature: 0.2, seed: 42 + i },
        (content) => {
          pieces += 1;
          lane.querySelector(".out").textContent = content;
          const seconds = (performance.now() - laneStarted) / 1000;
          lane.querySelector(".meta").textContent =
            `${pieces} pieces · rate ${(pieces / Math.max(seconds, 0.001)).toFixed(2)} tok/s`;
        },
        (first) => {
          ttft = first / 1000;
          lane.querySelector(".meta").textContent = `ttft ${ttft.toFixed(2)}s …`;
        }
      );
      const total = ((performance.now() - laneStarted) / 1000).toFixed(2);
      lane.querySelector(".meta").textContent =
        `${ttft === null ? "—" : ttft.toFixed(2)}s ttft · ${total}s total · ${result.finish ?? "?"}`;
      lane.classList.add("done");
    } catch (err) {
      lane.querySelector(".meta").textContent = `failed: ${err.message}`;
    }
  });

  await Promise.all(runs);
  el.burst.disabled = false;
  burstState.running = false;
}

el.burst.addEventListener("click", runBurst);
