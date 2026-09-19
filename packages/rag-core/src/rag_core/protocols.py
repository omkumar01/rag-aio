"""Protocol contracts for every rag-aio capability (ADR-0002).

Implementations depend on these abstractions, never on vendors or on each
other's concrete classes. All I/O-bearing operations are async; CPU-bound
implementations must offload to threads/processes themselves.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol, runtime_checkable

from .chunks import Chunk
from .context import Context
from .documents import Document
from .embeddings import Embedding, SparseEmbedding
from .generation import GenerationRequest, GenerationResult
from .queries import Query, QueryVariant
from .retrieval import RetrievalHit, RetrievalResult
from .types import SparseVector

# --- ingestion ----------------------------------------------------------------


@runtime_checkable
class DocumentLoader(Protocol):
    """Loads raw bytes and basic metadata from a source (file, URL, object store)."""

    async def load(self, source: str) -> tuple[bytes, dict[str, Any]]: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Parses raw source content into a canonical :class:`Document`."""

    def supported_types(self) -> set[str]:
        """File extensions or MIME types this parser handles (e.g. {'.pdf'})."""
        ...

    async def parse(self, source: str, data: bytes) -> Document: ...


@runtime_checkable
class OCRProcessor(Protocol):
    """Extracts text and layout from page images with provenance."""

    async def process_page(self, document_id: str, page: int, image: bytes) -> dict[str, Any]: ...


# --- chunking / tokenization --------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """Token counting/encoding aligned with a target model."""

    def count_tokens(self, text: str) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: Sequence[int]) -> str: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into chunks preserving provenance."""

    async def chunk(self, document: Document) -> list[Chunk]: ...


# --- embedding / indexing -----------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Dense text embedder with batching."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class SparseEmbedder(Protocol):
    """Sparse text embedder (BM25-style or learned sparse)."""

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]: ...


@runtime_checkable
class Indexer(Protocol):
    """Bulk-writes chunks + embeddings into storage backends."""

    async def index(
        self, embeddings: Sequence[Embedding], sparse: Sequence[SparseEmbedding]
    ) -> None: ...


# --- storage ------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """Vector search backend (Qdrant, FAISS, pgvector, ...)."""

    async def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict[str, Any],
        namespace: str | None = None,
    ) -> None: ...

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalHit]: ...

    async def delete(self, chunk_ids: Sequence[str], namespace: str | None = None) -> None: ...

    async def health(self) -> bool: ...


@runtime_checkable
class DocumentStore(Protocol):
    """CRUD for canonical documents (metadata + raw content references)."""

    async def put(self, document: Document) -> None: ...

    async def get(self, document_id: str) -> Document | None: ...

    async def delete(self, document_id: str) -> None: ...

    async def find_by_hash(self, content_hash: str) -> Document | None: ...


@runtime_checkable
class KeyValueStore(Protocol):
    """Simple namespaced key/value persistence."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes) -> None: ...

    async def delete(self, key: str) -> None: ...


@runtime_checkable
class Cache(Protocol):
    """Cache with TTL support; keys must embed configuration/content hashes."""

    async def get(self, key: str) -> bytes | None: ...

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...


# --- retrieval ----------------------------------------------------------------


@runtime_checkable
class Retriever(Protocol):
    """Executes one retrieval strategy (dense, sparse, metadata, ...)."""

    async def retrieve(self, query: Query) -> RetrievalResult: ...


@runtime_checkable
class HybridRetriever(Protocol):
    """Coordinates multiple retrievers concurrently and fuses their results."""

    async def retrieve(self, query: Query) -> RetrievalResult: ...


@runtime_checkable
class FusionStrategy(Protocol):
    """Merges result lists from multiple strategies (RRF, weighted, ...)."""

    def fuse(self, results: Sequence[RetrievalResult], top_k: int) -> RetrievalResult: ...


# --- reranking ----------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    """Second-stage ranking over a small candidate set."""

    async def rerank(
        self, query: Query, candidates: Sequence[RetrievalHit], top_k: int | None = None
    ) -> list[RetrievalHit]: ...


# --- query intelligence -------------------------------------------------------


@runtime_checkable
class QueryStrategy(Protocol):
    """Transforms a query (rewrite, expansion, HyDE, decomposition, ...)."""

    async def transform(self, query: Query) -> list[QueryVariant]: ...


# --- context engineering ------------------------------------------------------


@runtime_checkable
class ContextBuilder(Protocol):
    """Assembles a token-budgeted, citation-preserving context from hits."""

    async def build(
        self, query: Query, hits: Sequence[RetrievalHit], token_budget: int
    ) -> Context: ...


# --- generation ---------------------------------------------------------------


@runtime_checkable
class Generator(Protocol):
    """Provider-agnostic generation (normal + streaming)."""

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def stream(self, request: GenerationRequest) -> AsyncIterator[str]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Low-level provider client behind a Generator."""

    async def complete(self, request: GenerationRequest) -> GenerationResult: ...


# --- evaluation / observability ----------------------------------------------


@runtime_checkable
class EvaluationMetric(Protocol):
    """A named metric computed over predictions vs ground truth."""

    name: str

    def compute(
        self,
        ground_truth: Sequence[Sequence[str]],
        predictions: Sequence[Sequence[str]],
        k: int | None = None,
    ) -> float: ...


@runtime_checkable
class Observer(Protocol):
    """Observability hook for stage events (spans, metrics, audit logs)."""

    def record(self, event: str, attributes: dict[str, Any] | None = None) -> None: ...


__all__ = [
    "Cache",
    "Chunker",
    "ContextBuilder",
    "DocumentLoader",
    "DocumentParser",
    "DocumentStore",
    "Embedder",
    "EvaluationMetric",
    "FusionStrategy",
    "Generator",
    "HybridRetriever",
    "Indexer",
    "KeyValueStore",
    "LLMProvider",
    "OCRProcessor",
    "Observer",
    "QueryStrategy",
    "Reranker",
    "Retriever",
    "SparseEmbedder",
    "Tokenizer",
    "VectorStore",
]
