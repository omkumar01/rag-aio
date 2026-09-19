"""rag-generation: provider-agnostic LLM generation layer.

Adapters talk to providers over raw HTTP via httpx (no vendor SDKs). Each adapter
implements the :class:`rag_core.protocols.LLMProvider` contract (``complete``), and
most also expose streaming (``stream``) plus metadata helpers (``list_models``).
"""

from __future__ import annotations

from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .generation import (
    GenerationService,
    GeneratorRegistry,
    collect_stream,
    create_provider,
)
from .http import ProviderHttpClient
from .ollama_provider import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider
from .openai_provider import OpenAIProvider

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "GeminiProvider",
    "GenerationService",
    "GeneratorRegistry",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "ProviderHttpClient",
    "__version__",
    "collect_stream",
    "create_provider",
]
