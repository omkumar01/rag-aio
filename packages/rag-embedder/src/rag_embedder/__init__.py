"""rag-embedder: chunking, tokenization, embedding and indexing orchestration."""

from __future__ import annotations

from .chunking import (
    CHUNKERS,
    BaseChunker,
    ChunkerConfig,
    FixedChunker,
    ParentChildChunker,
    RecursiveChunker,
    SentenceChunker,
    StructuralChunker,
    TokenChunker,
    build_page_map,
    create_chunker,
)
from .embedding import (
    FastEmbedDense,
    FastEmbedSparse,
    MockEmbedder,
    MockSparseEmbedder,
    embed_many,
)
from .indexing import (
    IndexerConfig,
    IndexOutcome,
    IngestionIndexer,
    SparseCapableStore,
    VectorIndexer,
)
from .pipeline import EmbeddingPipeline, PipelineOutcome
from .tokenization import HFTokenizer, SimpleTokenizer

__version__ = "0.1.0"

__all__ = [
    "CHUNKERS",
    "BaseChunker",
    "ChunkerConfig",
    "EmbeddingPipeline",
    "FastEmbedDense",
    "FastEmbedSparse",
    "FixedChunker",
    "HFTokenizer",
    "IndexOutcome",
    "IndexerConfig",
    "IngestionIndexer",
    "MockEmbedder",
    "MockSparseEmbedder",
    "ParentChildChunker",
    "PipelineOutcome",
    "RecursiveChunker",
    "SentenceChunker",
    "SimpleTokenizer",
    "SparseCapableStore",
    "StructuralChunker",
    "TokenChunker",
    "VectorIndexer",
    "__version__",
    "build_page_map",
    "create_chunker",
    "embed_many",
]
