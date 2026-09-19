"""Tests for embedding strategies — no network, no model downloads."""

from __future__ import annotations

import math

import pytest
from rag_embedder.embedding import (
    FastEmbedDense,
    FastEmbedSparse,
    MockEmbedder,
    MockSparseEmbedder,
    embed_many,
)


class TestMockEmbedder:
    @pytest.mark.asyncio
    async def test_dim_and_normalize(self):
        emb = MockEmbedder(dim=64)
        vectors = await emb.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 64
        for v in vectors:
            norm = math.sqrt(sum(x * x for x in v))
            assert abs(norm - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_deterministic(self):
        emb = MockEmbedder(dim=32)
        v1 = await emb.embed(["same text"])
        v2 = await emb.embed(["same text"])
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts_differ(self):
        emb = MockEmbedder(dim=32)
        v1 = await emb.embed(["alpha"])
        v2 = await emb.embed(["beta"])
        assert v1 != v2

    @pytest.mark.asyncio
    async def test_empty(self):
        emb = MockEmbedder(dim=16)
        assert await emb.embed([]) == []


class TestMockSparseEmbedder:
    @pytest.mark.asyncio
    async def test_basic(self):
        emb = MockSparseEmbedder(vocab_size=1000)
        result = await emb.embed_sparse(["hello world hello"])
        assert len(result) == 1
        sv = result[0]
        assert len(sv.indices) == len(sv.values)
        assert all(i >= 0 for i in sv.indices)
        # "hello" appears twice
        # Find the index for "hello"
        import hashlib

        h_idx = int(hashlib.sha256(b"hello").hexdigest(), 16) % 1000
        if h_idx in sv.indices:
            pos = sv.indices.index(h_idx)
            assert sv.values[pos] == 2.0

    @pytest.mark.asyncio
    async def test_empty(self):
        emb = MockSparseEmbedder()
        assert await emb.embed_sparse([]) == []


class TestFastEmbedDense:
    def test_constructor_no_download(self):
        """Constructor must not instantiate the model."""
        created = []

        def fake_factory():
            created.append(True)
            return _FakeDenseModel(dim=128)

        emb = FastEmbedDense(model_name="test", model_factory=fake_factory)
        assert not created  # model not loaded yet
        assert emb._model is None

    def test_lazy_load(self):
        def fake_factory():
            return _FakeDenseModel(dim=128)

        emb = FastEmbedDense(model_name="test", model_factory=fake_factory)
        model = emb.ensure_model()
        assert isinstance(model, _FakeDenseModel)
        assert emb._model is model

    @pytest.mark.asyncio
    async def test_embed(self):
        def fake_factory():
            return _FakeDenseModel(dim=64)

        emb = FastEmbedDense(model_name="fake", model_factory=fake_factory, normalize=True)
        vectors = await emb.embed(["hello world", "foo bar"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 64
        for v in vectors:
            norm = math.sqrt(sum(x * x for x in v))
            assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_empty_embed(self):
        def fake_factory():
            return _FakeDenseModel(dim=64)

        emb = FastEmbedDense(model_name="fake", model_factory=fake_factory)
        assert await emb.embed([]) == []

    @property
    def dim_before_load(self):
        """dim is None or resolvable before model load."""
        emb = FastEmbedDense(
            model_name="nonexistent/xyz", model_factory=lambda: _FakeDenseModel(dim=99)
        )
        assert emb.dim is None or emb.dim == 99


class TestFastEmbedSparse:
    @pytest.mark.asyncio
    async def test_embed_sparse(self):
        def fake_factory():
            return _FakeSparseModel()

        emb = FastEmbedSparse(model_name="fake", model_factory=fake_factory)
        result = await emb.embed_sparse(["hello world test"])
        assert len(result) == 1
        sv = result[0]
        assert len(sv.indices) == len(sv.values)
        assert all(i >= 0 for i in sv.indices)


class TestEmbedMany:
    @pytest.mark.asyncio
    async def test_batching(self):
        emb = MockEmbedder(dim=16)
        texts = [f"text {i}" for i in range(10)]
        result = await embed_many(texts, emb, batch_size=3, max_concurrency=2)
        assert len(result) == 10
        for v in result:
            assert len(v) == 16

    @pytest.mark.asyncio
    async def test_empty(self):
        emb = MockEmbedder(dim=8)
        assert await embed_many([], emb) == []


class _FakeDenseModel:
    """Stand-in for fastembed.TextEmbedding in tests."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.model_name = "fake"

    def embed(self, documents, batch_size=256, parallel=None, **kwargs):
        import numpy as np

        for doc in documents:
            rng = np.random.RandomState(abs(hash(doc)) % (2**31))
            vec = rng.randn(self.dim).astype("float32")
            yield vec


class _FakeSparseModel:
    """Stand-in for fastembed.SparseTextEmbedding in tests."""

    def embed(self, documents, batch_size=256, parallel=None, **kwargs):
        import numpy as np
        from fastembed.sparse.sparse_embedding_base import SparseEmbedding

        for doc in documents:
            tokens = doc.lower().split()
            indices = [hash(t) % 1000 for t in tokens]
            values = [1.0] * len(tokens)
            yield SparseEmbedding(
                values=np.array(values, dtype=np.float64),
                indices=np.array(indices, dtype=np.int64),
            )
