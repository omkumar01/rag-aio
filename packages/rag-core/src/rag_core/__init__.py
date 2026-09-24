"""rag-core: canonical domain models and Protocol contracts for rag-aio.

This package is dependency-light (pydantic only). It defines the stable
contracts every other rag-aio module implements and consumes (ADR-0002).
"""

from .base import RagBaseModel
from .chunks import Chunk, ChunkMetadata
from .context import Citation, Context, ContextItem
from .documents import (
    BlockKind,
    BoundingBox,
    Document,
    DocumentAsset,
    DocumentMetadata,
    DocumentPage,
    PageBlock,
)
from .embeddings import Embedding, SparseEmbedding
from .errors import (
    BudgetExceededError,
    CacheError,
    ChunkingError,
    ConfigError,
    ContextError,
    CrawlError,
    EmbeddingError,
    EvaluationError,
    GenerationError,
    IngestionError,
    JobError,
    OperationTimeout,
    ParseError,
    ProviderError,
    ProviderUnavailableError,
    RagError,
    RateLimitError,
    RerankError,
    RetrievalError,
    StorageError,
    UnsupportedFormatError,
)
from .evaluation import EvaluationResult, MetricResult, QueryEvaluation
from .generation import FinishReason, GenerationRequest, GenerationResult, Message, Role, Usage
from .ids import config_hash, content_hash, new_id, stable_id
from .jobs import JobStatus, PipelineJob
from .models_info import HealthState, ModelInfo, ProviderInfo, ProviderKind
from .pipeline_config import (
    ContextStage,
    EmbeddingStage,
    GenerationStage,
    IngestionStage,
    PipelineConfig,
    RerankStage,
    RetrievalStage,
    StageConfig,
    pipeline_schema_version,
)
from .queries import Query, QueryVariant, QueryVariantKind
from .rerank import RerankHit
from .retrieval import RetrievalHit, RetrievalResult
from .serde import from_json, to_json
from .types import SparseVector

__version__ = "0.1.0"

__all__ = [
    "BlockKind",
    "BoundingBox",
    "BudgetExceededError",
    "CacheError",
    "Chunk",
    "ChunkMetadata",
    "ChunkingError",
    "Citation",
    "ConfigError",
    "Context",
    "ContextError",
    "ContextItem",
    "ContextStage",
    "CrawlError",
    "Document",
    "DocumentAsset",
    "DocumentMetadata",
    "DocumentPage",
    "Embedding",
    "EmbeddingError",
    "EmbeddingStage",
    "EvaluationError",
    "EvaluationResult",
    "FinishReason",
    "GenerationError",
    "GenerationRequest",
    "GenerationResult",
    "GenerationStage",
    "HealthState",
    "IngestionError",
    "IngestionStage",
    "JobError",
    "JobStatus",
    "Message",
    "MetricResult",
    "ModelInfo",
    "OperationTimeout",
    "PageBlock",
    "ParseError",
    "PipelineConfig",
    "PipelineJob",
    "ProviderError",
    "ProviderInfo",
    "ProviderKind",
    "ProviderUnavailableError",
    "Query",
    "QueryEvaluation",
    "QueryVariant",
    "QueryVariantKind",
    "RagBaseModel",
    "RagError",
    "RateLimitError",
    "RerankError",
    "RerankHit",
    "RerankStage",
    "RetrievalError",
    "RetrievalHit",
    "RetrievalResult",
    "RetrievalStage",
    "Role",
    "SparseEmbedding",
    "SparseVector",
    "StageConfig",
    "StorageError",
    "UnsupportedFormatError",
    "Usage",
    "config_hash",
    "content_hash",
    "from_json",
    "new_id",
    "pipeline_schema_version",
    "stable_id",
    "to_json",
]

# protocols are exported as a namespace to keep this __init__ readable
from . import protocols as protocols
