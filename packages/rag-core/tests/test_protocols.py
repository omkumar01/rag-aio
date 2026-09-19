"""Contract tests: every Protocol is runtime-checkable and structurally satisfiable
by a minimal reference implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
from rag_core.chunks import Chunk, ChunkMetadata
from rag_core.context import Context
from rag_core.documents import Document
from rag_core.embeddings import SparseEmbedding
from rag_core.generation import GenerationRequest, GenerationResult
from rag_core.queries import Query, QueryVariant
from rag_core.retrieval import RetrievalHit, RetrievalResult
from rag_core.types import SparseVector


def _doc(text: str = "t") -> Document:
    return Document(source_uri="test", text=text)


class Impls:
    """Minimal structural implementations used to prove the protocols."""

    from rag_core.protocols import (
        Cache,
        Chunker,
        ContextBuilder,
        DocumentParser,
        Embedder,
        EvaluationMetric,
        FusionStrategy,
        Generator,
        Indexer,
        QueryStrategy,
        Reranker,
        Retriever,
        SparseEmbedder,
        Tokenizer,
        VectorStore,
    )

    class Tokenizer(Tokenizer):
        def count_tokens(self, text: str) -> int:
            return len(text.split())

    class Chunker(Chunker):
        async def chunk(self, document: Document) -> list[Chunk]:
            return [
                Chunk(
                    document_id=document.id,
                    text=document.text,
                    index=0,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        document_hash=document.content_hash,
                        chunker="test",
                        chunker_version="0",
                    ),
                )
            ]

    class Embedder(Embedder):
        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[float(len(t)), 0.0] for t in texts]

    class SparseEmbedder(SparseEmbedder):
        async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
            return [SparseVector(indices=[0], values=[1.0]) for _ in texts]

    class VectorStore(VectorStore):
        async def upsert(
            self,
            chunk_id: str,
            vector: list[float],
            payload: dict[str, object],
            namespace: str | None = None,
        ) -> None: ...
        async def search(
            self,
            vector: list[float],
            top_k: int,
            filters: dict[str, object] | None = None,
            namespace: str | None = None,
        ) -> list[RetrievalHit]:
            return []

    class Retriever(Retriever):
        async def retrieve(self, query: Query) -> RetrievalResult:
            return RetrievalResult(query_id=query.id, hits=[], strategies=["test"])

    class FusionStrategy(FusionStrategy):
        def fuse(self, results: Sequence[RetrievalResult], top_k: int) -> RetrievalResult:
            return RetrievalResult(query_id="f", hits=[], strategies=["fused"])

    class Reranker(Reranker):
        async def rerank(
            self, query: Query, candidates: Sequence[RetrievalHit], top_k: int | None = None
        ) -> list[RetrievalHit]:
            return list(candidates)[: top_k or len(candidates)]

    class QueryStrategy(QueryStrategy):
        async def transform(self, query: Query) -> list[QueryVariant]:
            return []

    class ContextBuilder(ContextBuilder):
        async def build(
            self, query: Query, hits: Sequence[RetrievalHit], token_budget: int
        ) -> Context:
            return Context(items=[], token_budget=token_budget, strategy="test")

    class Generator(Generator):
        async def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(text="", model="m", finish_reason="stop")

        async def stream(self, request: GenerationRequest) -> AsyncIterator[str]:
            yield ""

    class Cache(Cache):
        async def get(self, key: str) -> bytes | None:
            return None

        async def set(self, key: str, value: bytes, ttl: float | None = None) -> None: ...

        async def delete(self, key: str) -> None: ...

    class DocumentParser(DocumentParser):
        def supported_types(self) -> set[str]:
            return {".txt"}

    class Indexer(Indexer):
        async def index(
            self,
            embeddings: Sequence[object],
            sparse: Sequence[SparseEmbedding],
        ) -> None: ...


@pytest.mark.parametrize(
    "protocol,impl_name",
    [
        ("Tokenizer", "Tokenizer"),
        ("Chunker", "Chunker"),
        ("Embedder", "Embedder"),
        ("SparseEmbedder", "SparseEmbedder"),
        ("VectorStore", "VectorStore"),
        ("Retriever", "Retriever"),
        ("FusionStrategy", "FusionStrategy"),
        ("Reranker", "Reranker"),
        ("QueryStrategy", "QueryStrategy"),
        ("ContextBuilder", "ContextBuilder"),
        ("Generator", "Generator"),
        ("Cache", "Cache"),
        ("DocumentParser", "DocumentParser"),
        ("Indexer", "Indexer"),
    ],
)
def test_protocol_runtime_checkable(protocol: str, impl_name: str) -> None:
    import rag_core.protocols as p

    proto = getattr(p, protocol)
    impl = getattr(Impls, impl_name)
    assert isinstance(impl(), proto), f"{impl_name} does not satisfy {protocol}"
