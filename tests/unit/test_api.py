"""API tests using the deterministic engine (no torch/model downloads).

Covers completions (sync + SSE), chat completions, health, stats, and
metrics endpoints.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from fluxllm.api.app import create_app
from fluxllm.config.settings import EngineSettings


@pytest.fixture
def client() -> TestClient:
    settings = EngineSettings(
        deterministic=True,
        vocab_size=32,
        num_blocks=16,
        block_size=4,
        max_batch_size=4,
        max_new_tokens=8,
        max_prompt_tokens=64,
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health_live(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_ready_reports_engine(client):
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["engine_running"] is True
    assert body["deterministic"] is True
    assert body["model"] == "fluxllm-simulated"


def test_completion_sync(client):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hello world", "max_tokens": 5, "temperature": 0.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert len(body["choices"]) == 1
    assert body["choices"][0]["finish_reason"] in ("length", "stop")
    assert body["usage"]["prompt_tokens"] == 2
    assert 1 <= body["usage"]["completion_tokens"] <= 5
    assert body["request_id"]


def test_completion_stream_sse(client):
    with client.stream(
        "POST", "/v1/completions", json={"prompt": "stream me", "max_tokens": 4, "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "data: [DONE]"
    chunks = [json.loads(e[6:]) for e in events[:-1]]
    token_chunks = [c for c in chunks if c["choices"][0]["text"]]
    assert len(token_chunks) == 4
    finish_chunks = [c for c in chunks if c["choices"][0]["finish_reason"] is not None]
    assert len(finish_chunks) == 1
    assert finish_chunks[0]["choices"][0]["finish_reason"] in ("length", "stop")


def test_chat_completions_sync(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["request_id"]


def test_chat_completions_stream(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert any(c["choices"][0]["delta"].get("content") for c in chunks)


def test_stats_endpoint(client):
    body = client.get("/stats").json()
    assert {"requests_queued", "requests_running", "kv_cache"} <= body.keys()


def test_metrics_endpoint_exposes_prometheus(client):
    client.post("/v1/completions", json={"prompt": "metrics test", "max_tokens": 2})
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert "fluxllm_requests_received_total" in text
    assert "fluxllm_generated_tokens_total" in text
    assert "fluxllm_kv_cache_utilization_ratio" in text
