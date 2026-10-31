"""Pydantic request/response schemas for the OpenAI-compatible API layer."""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(default="user")
    content: str


class BaseGenerationRequest(BaseModel):
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    stream: bool = False
    seed: int | None = None
    model: str | None = None  # accepted for OpenAI compatibility; engine decides

    def to_sampling(self) -> tuple[float, float, int]:
        return self.temperature, self.top_p, self.top_k


class CompletionsRequest(BaseGenerationRequest):
    prompt: str = Field(min_length=1, max_length=32_000)


class ChatCompletionsRequest(BaseGenerationRequest):
    messages: list[ChatMessage] = Field(min_length=1)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:24]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage
    # FluxLLM request id, exposed for correlation with logs and metrics.
    request_id: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: Usage
    request_id: str


class StatusInfo(BaseModel):
    state: str
    reason: str | None = None


class HealthInfo(BaseModel):
    status: Literal["ok", "degraded"]
    engine_running: bool
    model: str | None = None
    device: str | None = None
    deterministic: bool = False
