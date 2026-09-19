"""Provider and model configuration models.

These are the *control-plane* descriptions a user authors (YAML/JSON); they
hold endpoint config, auth *references*, declared model metadata, and routing
defaults. They carry no resolved credentials and perform no network I/O.
"""

from __future__ import annotations

from pydantic import Field, model_validator
from rag_core import ConfigError, RagBaseModel
from rag_core.models_info import ProviderKind

from .secrets import SecretRef

# Sensible default endpoints per provider kind. Local backends (LM Studio,
# vLLM, Ollama, llama.cpp) default to common localhost ports; cloud providers
# default to their public API roots.
DEFAULT_BASE_URLS: dict[ProviderKind, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "openai_compatible": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434",
    "vllm": "http://localhost:8000/v1",
    "llamacpp": "http://localhost:8080",
}


class ModelConfig(RagBaseModel):
    """Static metadata for one model exposed by a provider."""

    model_id: str
    roles: list[str] = Field(default_factory=list)
    context_window: int | None = None
    max_output_tokens: int | None = None
    embedding_dim: int | None = None
    capabilities: list[str] = Field(default_factory=list)
    temperature: float | None = None
    aliases: list[str] = Field(default_factory=list)


class ProviderConfig(RagBaseModel):
    """Configuration for a single LLM provider.

    ``base_url`` defaults to a sensible endpoint for ``kind`` when omitted.
    ``secret`` is an auth *reference* (env var / file / none), never a value.
    """

    name: str
    kind: ProviderKind
    base_url: str | None = None
    secret: SecretRef = Field(default_factory=SecretRef.none)
    models: list[ModelConfig] = Field(default_factory=list)
    default_models: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    timeout_s: float = 120.0
    max_retries: int = 2

    @model_validator(mode="after")
    def _finalize_and_validate(self) -> ProviderConfig:
        # Apply the kind-specific default base URL when none was supplied.
        if self.base_url is None:
            self.base_url = DEFAULT_BASE_URLS[self.kind]

        # Unique model ids within a provider.
        seen: set[str] = set()
        for model in self.models:
            if model.model_id in seen:
                raise ConfigError(
                    f"duplicate model_id {model.model_id!r} in provider {self.name!r}"
                )
            seen.add(model.model_id)

        # default_models entries must reference a declared model_id.
        for role, model_id in self.default_models.items():
            if model_id not in seen:
                raise ConfigError(
                    f"default_models[{role!r}]={model_id!r} references an unknown "
                    f"model_id (available: {sorted(seen) or 'none'}) in provider {self.name!r}"
                )
        return self


class ProvidersConfig(RagBaseModel):
    """Top-level providers section."""

    providers: list[ProviderConfig] = Field(default_factory=list)

    @classmethod
    def local_default(cls) -> ProvidersConfig:
        """Platform default profile (ADR-0004): LM Studio on localhost:1234.

        A single OpenAI-compatible provider with no secret, exposing generation,
        embedding, rerank, OCR/VLM, and query-rewrite models.
        """
        return cls(
            providers=[
                ProviderConfig(
                    name="lm-studio",
                    kind="openai_compatible",
                    models=[
                        ModelConfig(
                            model_id="qwen3-8b",
                            roles=["generate", "embed", "rerank", "ocr", "query_rewrite"],
                        ),
                    ],
                    default_models={
                        "generate": "qwen3-8b",
                        "embed": "qwen3-8b",
                        "rerank": "qwen3-8b",
                        "ocr": "qwen3-8b",
                        "query_rewrite": "qwen3-8b",
                    },
                )
            ]
        )


__all__ = [
    "DEFAULT_BASE_URLS",
    "ModelConfig",
    "ProviderConfig",
    "ProvidersConfig",
]
