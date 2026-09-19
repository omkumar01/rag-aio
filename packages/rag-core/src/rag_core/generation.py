"""Generation request/result models, provider-agnostic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from .base import RagBaseModel
from .ids import new_id

Role = Literal["system", "user", "assistant", "tool"]


class Message(RagBaseModel):
    role: Role
    content: str
    name: str | None = None


class Usage(RagBaseModel):
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0.0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class GenerationRequest(RagBaseModel):
    """A provider-agnostic generation request.

    ``model`` is optional: routing/fallback resolution happens upstream
    (rag-llm-provider / orchestrator), not in the request itself.
    """

    messages: list[Message]
    model: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: list[str] = Field(default_factory=list)
    json_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error", "cancelled"]


class GenerationResult(RagBaseModel):
    id: str = Field(default_factory=new_id)
    text: str
    model: str
    finish_reason: FinishReason
    usage: Usage | None = None
    tool_calls: list[dict[str, Any]] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
