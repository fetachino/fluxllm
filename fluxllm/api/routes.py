"""HTTP routes for the FluxLLM API.

Thin by design: this layer validates payloads, maps engine streams to
OpenAI-style JSON/SSE, and exposes health + metrics. All inference logic
lives in the engine package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from fluxllm.api.schemas import (
    ChatChoice,
    ChatCompletionResponse,
    ChatCompletionsRequest,
    ChatMessage,
    CompletionChoice,
    CompletionResponse,
    CompletionsRequest,
    HealthInfo,
    Usage,
)
from fluxllm.engine.engine import EngineFullError, RequestHandle
from fluxllm.engine.lifecycle import RequestState
from fluxllm.sampling.config import SamplingParams

logger = logging.getLogger(__name__)
router = APIRouter()

_FINISH_REASON_MAP = {
    "length": "length",
    "eos_token": "stop",
    "stop_token": "stop",
    "stop_string": "stop",
}
_STATE_MAP = {
    "completed": "completed",
    "cancelled": "length",  # client disconnected; response cut short anyway
    "failed": "error",
    "error": "error",
}


def _build_sampling(request: CompletionsRequest | ChatCompletionsRequest) -> SamplingParams:
    return SamplingParams(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        seed=request.seed,
    )


def _usage(handle: RequestHandle) -> Usage:
    return Usage(
        prompt_tokens=handle.request.num_prompt_tokens,
        completion_tokens=handle.request.num_generated_tokens,
        total_tokens=handle.request.num_prompt_tokens + handle.request.num_generated_tokens,
    )


def _completion_response(handle: RequestHandle, text: str) -> CompletionResponse:
    request = handle.request
    finish = (
        _FINISH_REASON_MAP.get(request.completion_reason.value, "stop")
        if request.completion_reason
        else "error"
    )
    return CompletionResponse(
        model="fluxllm",
        choices=[CompletionChoice(text=text, finish_reason=finish)],
        usage=_usage(handle),
        request_id=request.request_id,
    )


@router.post("/v1/completions", response_model=None)
async def completions(
    request_data: CompletionsRequest, request: Request
) -> CompletionResponse | StreamingResponse:
    engine = request.app.state.engine
    try:
        handle = engine.submit(request_data.prompt, _build_sampling(request_data))
    except EngineFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request_data.stream:
        return StreamingResponse(
            _stream_completion_sse(request.app, handle, prompt_text=request_data.prompt),
            media_type="text/event-stream",
            headers={"X-Request-Id": handle.request.request_id, "Cache-Control": "no-cache"},
        )
    text = await _collect_text(request.app, handle)
    final = _completion_response(handle, text)
    return final


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    request_data: ChatCompletionsRequest, request: Request
) -> ChatCompletionResponse | StreamingResponse:
    engine = request.app.state.engine
    prompt = "\n".join(f"{message.role}: {message.content}" for message in request_data.messages)
    try:
        handle = engine.submit(prompt, _build_sampling(request_data))
    except EngineFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if request_data.stream:
        return StreamingResponse(
            _stream_chat_sse(request.app, handle),
            media_type="text/event-stream",
            headers={"X-Request-Id": handle.request.request_id, "Cache-Control": "no-cache"},
        )
    text = await _collect_text(request.app, handle)
    if handle.request.state in (RequestState.FAILED.value, RequestState.CANCELLED.value):
        raise HTTPException(status_code=502, detail="generation did not complete")
    response = ChatCompletionResponse(
        model="fluxllm",
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=text))],
        usage=_usage(handle),
        request_id=handle.request.request_id,
    )
    return response


async def _collect_text(app: FastAPI, handle: RequestHandle) -> str:
    """Await a full non-streaming generation."""
    started = time.perf_counter()
    async for _event in handle.stream():
        pass
    logger.info(
        "request %s completed in %.3fs state=%s reason=%s",
        handle.request.request_id,
        time.perf_counter() - started,
        handle.request.state.value,
        handle.request.completion_reason.value if handle.request.completion_reason else None,
    )
    if handle.request.state is RequestState.FAILED:
        raise HTTPException(status_code=502, detail=handle.request.error or "generation failed")
    if handle.request.state is not RequestState.COMPLETED:
        raise HTTPException(status_code=502, detail="generation did not complete")
    return app.state.detokenize(handle.request.generated_token_ids)


async def _stream_completion_sse(
    app: FastAPI, handle: RequestHandle, prompt_text: str
) -> AsyncIterator[str]:
    """Yield OpenAI-style SSE chunks for /v1/completions."""
    created = int(time.time())
    completion_id = f"cmpl-{handle.request.request_id[:24]}"
    previous = ""
    try:
        async for event in handle.stream():
            if event.type == "token":
                full = app.state.detokenize(handle.request.generated_token_ids)
                piece = full[len(previous) :] if full.startswith(previous) else full
                previous = full
                if piece:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": "fluxllm",
                            "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                        }
                    )
            elif event.type == "status":
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": "fluxllm",
                        "choices": [
                            {
                                "index": 0,
                                "text": "",
                                "finish_reason": _map_reason(event.reason),
                            }
                        ],
                    }
                )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        handle.cancel()
        raise


def _map_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    return _FINISH_REASON_MAP.get(reason, _STATE_MAP.get(reason, "stop"))


async def _stream_chat_sse(app: FastAPI, handle: RequestHandle) -> AsyncIterator[str]:
    created = int(time.time())
    completion_id = f"chatcmpl-{handle.request.request_id[:24]}"
    previous = ""
    try:
        async for event in handle.stream():
            if event.type == "token":
                full = app.state.detokenize(handle.request.generated_token_ids)
                piece = full[len(previous) :] if full.startswith(previous) else full
                previous = full
                if piece:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": "fluxllm",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": piece},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            elif event.type == "status":
                yield _sse(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": "fluxllm",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": _map_reason(event.reason),
                            }
                        ],
                    }
                )
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        handle.cancel()
        raise


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/ready")
async def ready(request: Request) -> HealthInfo:
    state = request.app.state
    engine = getattr(state, "engine", None)
    return HealthInfo(
        status="ok" if engine is not None else "degraded",
        engine_running=engine is not None,
        model=getattr(state, "model_name", None),
        device=getattr(state, "device", None),
        deterministic=getattr(state, "deterministic", False),
    )


@router.get("/stats")
async def stats(request: Request) -> dict:
    return request.app.state.engine.stats()


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    return Response(
        content=request.app.state.metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
