"""Tests for pipeline.py end-to-end — no network, no model downloads."""

from __future__ import annotations

import pytest
from rag_core.documents import Document
from rag_core.ids import config_hash
from rag_embedder.chunking import ChunkerConfig, create_chunker
from rag_embedder.embedding import MockEmbedder, MockSparseEmbedder
from rag_embedder.pipeline import EmbeddingPipeline, PipelineOutcome


def _make_document(text: str | None = None) -> Document:
    if text is None:
        text = " ".join(f"This is test paragraph number {i} with some content." for i in range(30))
    return Document(source_uri="test://pipeline", text=text, id="doc-pipeline")


class TestPipeline:
    @pytest.fixture
    def pipeline(self, tokenizer, fake_store_cls):
        config = ChunkerConfig(strategy="recursive", chunk_size=30, overlap=5, min_chunk_size=1)
        chunker = create_chunker(config, tokenizer)
        dense = MockEmbedder(dim=16)
        sparse = MockSparseEmbedder(vocab_size=1000)
        return EmbeddingPipeline(
            chunker=chunker,
            embedder=dense,
            sparse_embedder=sparse,
            store=fake_store_cls(),
            config=None,
        )

    @pytest.mark.asyncio
    async def test_process_end_to_end(self, pipeline, tokenizer, fake_store_cls):
        doc = _make_document()
        store = fake_store_cls()
        outcome = await pipeline.process(doc, store=store)
        assert isinstance(outcome, PipelineOutcome)
        assert outcome.chunks > 0
        assert outcome.chunks_indexed == outcome.chunks
        assert outcome.dims == 16
        assert outcome.model == "mock-dense"
        assert outcome.sparse_model == "mock-sparse"
        assert len(store.points) == outcome.chunks_indexed
        assert len(store.sparse_points) == outcome.chunks_indexed

    @pytest.mark.asyncio
    async def test_no_sparse_embedder(self, tokenizer, fake_store_cls):
        chunker = create_chunker(
            ChunkerConfig(strategy="recursive", chunk_size=30, overlap=5, min_chunk_size=1),
            tokenizer,
        )
        pipeline = EmbeddingPipeline(
            chunker=chunker,
            embedder=MockEmbedder(dim=8),
            sparse_embedder=None,
            store=fake_store_cls(),
        )
        doc = _make_document()
        store = fake_store_cls()
        outcome = await pipeline.process(doc, store=store)
        assert outcome.sparse_model is None
        assert len(store.sparse_points) == 0
        assert len(store.points) == outcome.chunks_indexed


class TestShouldReindex:
    def test_first_time_reindex(self, tokenizer):
        doc = _make_document()
        seen: dict[str, str] = {}
        ch = config_hash("config-v1")
        assert EmbeddingPipeline.should_reindex(doc, ch, seen) is True
        assert doc.id in seen

    def test_skip_if_unchanged(self, tokenizer):
        doc = _make_document()
        seen: dict[str, str] = {}
        ch = config_hash("config-v1")
        EmbeddingPipeline.should_reindex(doc, ch, seen)
        assert EmbeddingPipeline.should_reindex(doc, ch, seen) is False

    def test_reindex_on_content_change(self, tokenizer):
        doc1 = _make_document("content A")
        doc2 = _make_document("content B")
        seen: dict[str, str] = {}
        ch = config_hash("config-v1")
        EmbeddingPipeline.should_reindex(doc1, ch, seen)
        assert EmbeddingPipeline.should_reindex(doc2, ch, seen) is True

    def test_reindex_on_config_change(self, tokenizer):
        doc = _make_document()
        seen: dict[str, str] = {}
        ch1 = config_hash("config-v1")
        ch2 = config_hash("config-v2")
        EmbeddingPipeline.should_reindex(doc, ch1, seen)
        assert EmbeddingPipeline.should_reindex(doc, ch2, seen) is True


class TestPipelineChunkerName:
    @pytest.mark.asyncio
    async def test_chunker_name_in_metadata(self, tokenizer):
        chunker = create_chunker(
            ChunkerConfig(strategy="fixed", chunk_size=80, overlap=10, min_chunk_size=1),
            tokenizer,
        )
        doc = _make_document()
        chunks = await chunker.chunk(doc)
        assert all(c.metadata.chunker == "fixed" for c in chunks)
