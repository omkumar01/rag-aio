"""Provider and model metadata (control-plane information, no secrets)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from .base import RagBaseModel

ProviderKind = Literal[
    "openai", "anthropic", "gemini", "openai_compatible", "ollama", "vllm", "llamacpp"
]
HealthState = Literal["unknown", "healthy", "degraded", "down"]


class ModelInfo(RagBaseModel):
    """Static metadata about one model exposed by a provider."""

    model_id: str
    provider: str
    display_name: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    context_window: int | None = Field(default=None, ge=0)
    max_output_tokens: int | None = Field(default=None, ge=0)
    embedding_dim: int | None = Field(default=None, ge=0)
    pricing: dict[str, float] = Field(default_factory=dict)


class ProviderInfo(RagBaseModel):
    """Provider endpoint configuration.

    Authentication is by reference only (``auth_ref`` names an environment
    variable, secret file, or keyring entry). Raw credentials never appear in
    this model, in API responses, logs, or the UI.
    """

    name: str
    kind: ProviderKind
    base_url: str | None = None
    auth_ref: str | None = None
    models: list[ModelInfo] = Field(default_factory=list)
    default_models: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    health: HealthState = "unknown"
