"""Tests for chunking strategies — no network, no model downloads."""

from __future__ import annotations

import itertools

import pytest
from rag_core.chunks import Chunk
from rag_core.protocols import Tokenizer
from rag_embedder.chunking import (
    ChunkerConfig,
    FixedChunker,
    ParentChildChunker,
    RecursiveChunker,
    SentenceChunker,
    StructuralChunker,
    TokenChunker,
    create_chunker,
)
from rag_embedder.tokenization import SimpleTokenizer

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestRegistry:
    def test_create_chunker_recursive_default(self, tokenizer: Tokenizer):
        config = ChunkerConfig()
        chunker = create_chunker(config, tokenizer)
        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunker_name == "recursive"

    def test_create_chunker_all_strategies(self, tokenizer: Tokenizer):
        for strategy, expected in [
            ("fixed", FixedChunker),
            ("sentence", SentenceChunker),
            ("token", TokenChunker),
            ("recursive", RecursiveChunker),
            ("structural", StructuralChunker),
            ("parent_child", ParentChildChunker),
        ]:
            config = ChunkerConfig(strategy=strategy)
            chunker = create_chunker(config, tokenizer)
            assert isinstance(chunker, expected)

    def test_unknown_strategy_raises(self, tokenizer: Tokenizer):
        config = ChunkerConfig(strategy="recursive")
        # Patch the config to use an unknown strategy
        config = ChunkerConfig.model_construct(strategy="bogus")
        with pytest.raises(ValueError, match="unknown chunker strategy"):
            create_chunker(config, tokenizer)


# --------------------------------------------------------------------------- #
# Fixed chunker
# --------------------------------------------------------------------------- #


class TestFixedChunker:
    @pytest.mark.asyncio
    async def test_overlap_math(self, tokenizer: SimpleTokenizer, long_text: str):
        from rag_core.documents import Document

        doc = Document(source_uri="test://fixed", text=long_text, id="doc-fixed")
        config = ChunkerConfig(strategy="fixed", chunk_size=100, overlap=20, min_chunk_size=1)
        chunker = FixedChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)

        assert len(chunks) >= 3
        # Each chunk text ≤ chunk_size characters
        for chunk in chunks:
            assert len(chunk.text) <= 100
        # Consecutive overlap == 20 chars
        for a, b in itertools.pairwise(chunks):
            assert b.text[:20] == a.text[-20:]

    @pytest.mark.asyncio
    async def test_produces_valid_chunks(self, tokenizer: SimpleTokenizer, long_text: str):
        from rag_core.documents import Document

        doc = Document(source_uri="test://fixed2", text=long_text, id="doc-fixed2")
        config = ChunkerConfig(strategy="fixed", chunk_size=80, overlap=10, min_chunk_size=1)
        chunker = FixedChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.text
            assert chunk.metadata.chunker == "fixed"
            assert chunk.metadata.chunker_version == "0.1.0"
            assert chunk.metadata.document_id == doc.id
            assert chunk.id  # non-empty deterministic id


# --------------------------------------------------------------------------- #
# Sentence chunker
# --------------------------------------------------------------------------- #


class TestSentenceChunker:
    @pytest.mark.asyncio
    async def test_budget_respected(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = " ".join(f"This is sentence number {i}. It has some content." for i in range(50))
        doc = Document(source_uri="test://sent", text=text, id="doc-sent")
        config = ChunkerConfig(strategy="sentence", chunk_size=30, overlap=1, min_chunk_size=1)
        chunker = SentenceChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert tokenizer.count_tokens(chunk.text) <= 30

    @pytest.mark.asyncio
    async def test_sentence_boundaries(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = "First sentence here. Second sentence follows. Third one too."
        doc = Document(source_uri="test://sent2", text=text, id="doc-sent2")
        config = ChunkerConfig(strategy="sentence", chunk_size=100, overlap=0, min_chunk_size=1)
        chunker = SentenceChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert "First sentence" in chunks[0].text


# --------------------------------------------------------------------------- #
# Token chunker
# --------------------------------------------------------------------------- #


class TestTokenChunker:
    @pytest.mark.asyncio
    async def test_token_counts_le_budget(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = "word " * 200  # 200 tokens
        doc = Document(source_uri="test://tok", text=text, id="doc-tok")
        config = ChunkerConfig(strategy="token", chunk_size=50, overlap=10, min_chunk_size=1)
        chunker = TokenChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert tokenizer.count_tokens(chunk.text) <= 50

    @pytest.mark.asyncio
    async def test_overlap(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = ". ".join(["hello world"] * 100)
        doc = Document(source_uri="test://tok2", text=text, id="doc-tok2")
        config = ChunkerConfig(strategy="token", chunk_size=20, overlap=5, min_chunk_size=1)
        chunker = TokenChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2


# --------------------------------------------------------------------------- #
# Recursive chunker (default)
# --------------------------------------------------------------------------- #


class TestRecursiveChunker:
    @pytest.mark.asyncio
    async def test_no_content_lost(self, tokenizer: SimpleTokenizer, long_text: str):
        from rag_core.documents import Document

        doc = Document(source_uri="test://rec", text=long_text, id="doc-rec")
        config = ChunkerConfig(strategy="recursive", chunk_size=60, overlap=10, min_chunk_size=1)
        chunker = RecursiveChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) > 1
        reconstructed = "".join(c.text for c in chunks)
        # No content lost: original is substring of reconstruction (allowing overlap)
        assert long_text[0:50] in reconstructed
        assert long_text[-50:] in reconstructed
        # Token budget respected (allowing overlap tokens)
        for chunk in chunks:
            assert tokenizer.count_tokens(chunk.text) <= 60 + 10  # overlap allowance

    @pytest.mark.asyncio
    async def test_is_default(self, tokenizer: SimpleTokenizer, long_text: str):
        from rag_core.documents import Document

        doc = Document(source_uri="test://rec2", text=long_text, id="doc-rec2")
        config = ChunkerConfig()  # default recursive, chunk_size=512
        chunker = create_chunker(config, tokenizer)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 1
        assert chunker.chunker_name == "recursive"


# --------------------------------------------------------------------------- #
# Structural chunker
# --------------------------------------------------------------------------- #


class TestStructuralChunker:
    @pytest.mark.asyncio
    async def test_markdown_headings(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = (
            "# Introduction\nThis is the intro.\nSome intro text.\n\n"
            "## Background\nBackground details here.\nMore background.\n\n"
            "## Conclusion\nThe conclusion goes here."
        )
        doc = Document(source_uri="test://struct", text=text, id="doc-struct")
        config = ChunkerConfig(strategy="structural", chunk_size=512, overlap=20, min_chunk_size=1)
        chunker = StructuralChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 2
        section_paths = [c.metadata.section_path for c in chunks]
        # At least one chunk should have "Introduction" in its path
        joined = [p for p in section_paths if p]
        assert any("Introduction" in p for p in joined)
        assert any("Background" in p for p in joined)

    @pytest.mark.asyncio
    async def test_html_headings(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = "<h1>Title One</h1>\nContent for title one.\n\n<h2>Subtitle</h2>\nSub content here."
        doc = Document(source_uri="test://struct2", text=text, id="doc-struct2")
        config = ChunkerConfig(strategy="structural", chunk_size=512, overlap=0, min_chunk_size=1)
        chunker = StructuralChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        assert len(chunks) >= 1
        paths = [c.metadata.section_path for c in chunks]
        assert any("Title One" in p for p in paths)


# --------------------------------------------------------------------------- #
# Parent-child chunker
# --------------------------------------------------------------------------- #


class TestParentChildChunker:
    @pytest.mark.asyncio
    async def test_children_reference_parents(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = " ".join(f"This is test paragraph number {i} with some words." for i in range(100))
        doc = Document(source_uri="test://pc", text=text, id="doc-pc")
        # Small chunk sizes so both parents and children are created
        config = ChunkerConfig(strategy="parent_child", chunk_size=20, overlap=4, min_chunk_size=1)
        chunker = ParentChildChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        # Should have both parent and child chunks
        has_parent = any(c.metadata.parent_chunk_id is None for c in chunks)
        has_child = any(c.metadata.parent_chunk_id is not None for c in chunks)
        assert has_parent
        assert has_child

    @pytest.mark.asyncio
    async def test_parents_before_children(self, tokenizer: SimpleTokenizer):
        from rag_core.documents import Document

        text = " ".join(f"Word word word word word number {i}." for i in range(200))
        doc = Document(source_uri="test://pc2", text=text, id="doc-pc2")
        config = ChunkerConfig(strategy="parent_child", chunk_size=15, overlap=4, min_chunk_size=1)
        chunker = ParentChildChunker(tokenizer, config)
        chunks = await chunker.chunk(doc)
        # Find first child chunk
        first_child_idx = next(
            i for i, c in enumerate(chunks) if c.metadata.parent_chunk_id is not None
        )
        # All chunks before first child should be parents (no parent_chunk_id)
        for c in chunks[:first_child_idx]:
            assert c.metadata.parent_chunk_id is None
        # Parent ids referenced by children should exist in the chunk list
        parent_ids = {c.id for c in chunks if c.metadata.parent_chunk_id is None}
        for c in chunks[first_child_idx:]:
            assert c.metadata.parent_chunk_id in parent_ids


# --------------------------------------------------------------------------- #
# Deterministic ids & page map
# --------------------------------------------------------------------------- #


class TestDeterminism:
    @pytest.mark.asyncio
    async def test_same_input_same_ids(self, tokenizer: SimpleTokenizer, long_text: str):
        from rag_core.documents import Document

        doc = Document(source_uri="test://det", text=long_text, id="doc-det")
        config = ChunkerConfig(strategy="recursive", chunk_size=60, overlap=10, min_chunk_size=1)
        chunker1 = RecursiveChunker(tokenizer, config)
        chunker2 = RecursiveChunker(tokenizer, config)
        chunks1 = await chunker1.chunk(doc)
        chunks2 = await chunker2.chunk(doc)
        ids1 = [c.id for c in chunks1]
        ids2 = [c.id for c in chunks2]
        assert ids1 == ids2


class TestPageMap:
    @pytest.mark.asyncio
    async def test_page_numbers(self, tokenizer: SimpleTokenizer, multi_page_doc):
        config = ChunkerConfig(strategy="recursive", chunk_size=15, overlap=5, min_chunk_size=1)
        chunker = RecursiveChunker(tokenizer, config)
        chunks = await chunker.chunk(multi_page_doc)
        assert len(chunks) > 0
        all_pages = set()
        for chunk in chunks:
            all_pages.update(chunk.metadata.page_numbers)
        # Should see both pages
        assert 1 in all_pages
        assert 2 in all_pages

    def test_build_page_map_multi(self, multi_page_doc):
        from rag_embedder.chunking.base import build_page_map

        pm = build_page_map(multi_page_doc)
        assert len(pm) == 2
        assert pm[0][2] == 1  # page 1
        assert pm[1][2] == 2  # page 2
        assert pm[0][0] == 0  # starts at 0

    def test_build_page_map_empty(self, small_doc):
        from rag_core.documents import Document
        from rag_embedder.chunking.base import build_page_map

        doc = Document(source_uri="test://empty", text="hello", id="doc-empty")
        assert build_page_map(doc) == []
