"""Tests for indexing.py — no network, no model downloads."""

from __future__ import annotations

import pytest
from rag_core.chunks import Chunk, ChunkMetadata
from rag_core.embeddings import Embedding, SparseEmbedding
from rag_embedder.embedding import MockEmbedder, MockSparseEmbedder
from rag_embedder.indexing import (
    IndexerConfig,
    IngestionIndexer,
    SparseCapableStore,
    VectorIndexer,
)


class PlainStore:
    """A VectorStore that does NOT support sparse upserts."""

    def __init__(self) -> None:
        self.points: dict[str, tuple[list[float], dict]] = {}

    async def upsert(self, chunk_id, vector, payload, namespace=None):
        self.points[chunk_id] = (vector[:], dict(payload))

    async def search(self, vector, top_k, filters=None, namespace=None):
        return []

    async def delete(self, chunk_ids, namespace=None):
        for cid in chunk_ids:
            self.points.pop(cid, None)

    async def health(self):
        return True


def _make_chunk(text: str = "test chunk", doc_id: str = "doc-1", idx: int = 0) -> Chunk:
    return Chunk(
        document_id=doc_id,
        text=text,
        index=idx,
        metadata=ChunkMetadata(
            document_id=doc_id,
            document_hash="hash123",
            chunker="recursive",
            chunker_version="0.1.0",
            page_numbers=[1],
            char_start=0,
            char_end=len(text),
            token_count=len(text.split()),
        ),
    )


def _make_embedding(chunk_id: str, dim: int = 64) -> Embedding:
    return Embedding(
        chunk_id=chunk_id,
        vector=[0.1] * dim,
        model="mock",
        dimension=dim,
        normalized=True,
    )


def _make_sparse(chunk_id: str) -> SparseEmbedding:
    return SparseEmbedding(
        chunk_id=chunk_id,
        indices=[0, 5, 10],
        values=[1.0, 0.5, 0.3],
        model="mock-sparse",
    )


class TestVectorIndexer:
    @pytest.mark.asyncio
    async def test_basic_upsert(self, fake_store):
        emb = _make_embedding("chunk-1", dim=32)
        indexer = VectorIndexer(fake_store)
        await indexer.index([emb], [])
        assert "chunk-1" in fake_store.points
        vec, payload, _ns = fake_store.points["chunk-1"]
        assert vec == [0.1] * 32
        assert payload["chunk_id"] == "chunk-1"

    @pytest.mark.asyncio
    async def test_sparse_upsert_with_capable_store(self, fake_store):
        emb = _make_embedding("chunk-1", dim=32)
        sp = _make_sparse("chunk-1")
        indexer = VectorIndexer(fake_store)
        await indexer.index([emb], [sp])
        assert "chunk-1" in fake_store.sparse_points
        indices, values, _ = fake_store.sparse_points["chunk-1"]
        assert indices == [0, 5, 10]
        assert values == [1.0, 0.5, 0.3]

    @pytest.mark.asyncio
    async def test_sparse_skipped_without_capability(self):
        store = PlainStore()
        emb = _make_embedding("chunk-1", dim=32)
        sp = _make_sparse("chunk-1")
        indexer = VectorIndexer(store)
        await indexer.index([emb], [sp])
        assert "chunk-1" in store.points
        assert not hasattr(store, "sparse_points")

    @pytest.mark.asyncio
    async def test_metadata_provider(self, fake_store):
        emb = _make_embedding("chunk-1", dim=16)
        indexer = VectorIndexer(
            fake_store,
            metadata_provider=lambda cid: {"custom": "payload", "chunk_id": cid},
        )
        await indexer.index([emb], [])
        _, payload, _ = fake_store.points["chunk-1"]
        assert payload["custom"] == "payload"

    @pytest.mark.asyncio
    async def test_batch_size(self, fake_store):
        embeddings = [_make_embedding(f"c-{i}", dim=8) for i in range(5)]
        indexer = VectorIndexer(
            fake_store,
            config=IndexerConfig(batch_size=2, max_concurrency=1),
        )
        await indexer.index(embeddings, [])
        for i in range(5):
            assert f"c-{i}" in fake_store.points

    @pytest.mark.asyncio
    async def test_empty(self, fake_store):
        indexer = VectorIndexer(fake_store)
        await indexer.index([], [])
        assert len(fake_store.points) == 0

    @pytest.mark.asyncio
    async def test_namespace(self, fake_store):
        emb = _make_embedding("c-1", dim=4)
        indexer = VectorIndexer(fake_store, config=IndexerConfig(namespace="ns1"))
        await indexer.index([emb], [])
        _, _, ns = fake_store.points["c-1"]
        assert ns == "ns1"


class TestIngestionIndexer:
    @pytest.mark.asyncio
    async def test_index_document(self, fake_store):
        from rag_core.documents import Document

        doc = Document(source_uri="test://ind", text="hello world foo bar", id="doc-ind")
        chunks = [
            _make_chunk("hello world", doc.id, 0),
            _make_chunk("foo bar", doc.id, 1),
        ]
        dense = MockEmbedder(dim=16)
        sparse = MockSparseEmbedder(vocab_size=1000)
        indexer = IngestionIndexer()
        outcome = await indexer.index_document(doc, chunks, dense, sparse, fake_store)
        assert outcome.chunks_indexed == 2
        assert outcome.dims == 16
        assert outcome.model == "mock-dense"
        assert outcome.sparse_model == "mock-sparse"
        assert len(fake_store.points) == 2
        assert len(fake_store.sparse_points) == 2

    @pytest.mark.asyncio
    async def test_index_document_no_sparse(self, fake_store):
        from rag_core.documents import Document

        doc = Document(source_uri="test://ind2", text="hello world", id="doc-ind2")
        chunks = [_make_chunk("hello world", doc.id, 0)]
        dense = MockEmbedder(dim=8)
        indexer = IngestionIndexer()
        outcome = await indexer.index_document(doc, chunks, dense, None, fake_store)
        assert outcome.chunks_indexed == 1
        assert outcome.sparse_model is None
        assert len(fake_store.points) == 1


class TestSparseCapableStore:
    def test_protocol_exists(self):
        assert hasattr(SparseCapableStore, "upsert_sparse")

    def test_fake_store_is_capable(self, fake_store):
        assert isinstance(fake_store, SparseCapableStore)

    def test_plain_store_not_capable(self):
        store = PlainStore()
        assert not isinstance(store, SparseCapableStore)
