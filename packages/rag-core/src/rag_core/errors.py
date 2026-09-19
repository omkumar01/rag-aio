"""rag-core error taxonomy.

Every module raises subclasses of :class:`RagError` carrying a stable machine
code and structured details. Exceptions that should reach the orchestrator are
never swallowed silently.
"""

from __future__ import annotations

from typing import Any


class RagError(Exception):
    """Base class for all rag-aio errors."""

    default_code: str = "rag_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code or self.default_code
        self.details: dict[str, Any] = details or {}


def _err(name: str, code: str, base: type[RagError] = RagError) -> type[RagError]:
    """Define an error class with a stable default code."""
    return type(name, (base,), {"default_code": code, "__module__": __name__})


# --- configuration ---
ConfigError = _err("ConfigError", "config_error")

# --- ingestion / documents ---
IngestionError = _err("IngestionError", "ingestion_error")
ParseError = _err("ParseError", "parse_error", IngestionError)
UnsupportedFormatError = _err("UnsupportedFormatError", "unsupported_format", IngestionError)
CrawlError = _err("CrawlError", "crawl_error", IngestionError)
OCRError = _err("OCRError", "ocr_error", IngestionError)
ChunkingError = _err("ChunkingError", "chunking_error", IngestionError)

# --- embedding / indexing ---
EmbeddingError = _err("EmbeddingError", "embedding_error")
IndexError_ = _err("IndexError_", "index_error")

# --- storage ---
StorageError = _err("StorageError", "storage_error")
CacheError = _err("CacheError", "cache_error")

# --- retrieval ---
RetrievalError = _err("RetrievalError", "retrieval_error")
FusionError = _err("FusionError", "fusion_error", RetrievalError)
RerankError = _err("RerankError", "rerank_error")
ContextError = _err("ContextError", "context_error")

# --- providers / generation ---
ProviderError = _err("ProviderError", "provider_error")
ProviderUnavailableError = _err("ProviderUnavailableError", "provider_unavailable", ProviderError)
RateLimitError = _err("RateLimitError", "rate_limited", ProviderError)
GenerationError = _err("GenerationError", "generation_error", ProviderError)

# --- execution ---
OperationTimeout = _err("OperationTimeout", "timeout")
BudgetExceededError = _err("BudgetExceededError", "budget_exceeded")
JobError = _err("JobError", "job_error")
EvaluationError = _err("EvaluationError", "evaluation_error")

__all__ = [
    "BudgetExceededError",
    "CacheError",
    "ChunkingError",
    "ConfigError",
    "ContextError",
    "CrawlError",
    "EmbeddingError",
    "EvaluationError",
    "FusionError",
    "GenerationError",
    "IndexError_",
    "IngestionError",
    "JobError",
    "OCRError",
    "OperationTimeout",
    "ParseError",
    "ProviderError",
    "ProviderUnavailableError",
    "RagError",
    "RateLimitError",
    "RerankError",
    "RetrievalError",
    "StorageError",
    "UnsupportedFormatError",
]
