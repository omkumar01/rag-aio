"""Dense and sparse embedding strategies.

* :class:`FastEmbedDense` — production dense embedder backed by ``fastembed``.
* :class:`FastEmbedSparse` — production sparse embedder (BM25-style).
* :class:`MockEmbedder` / :class:`MockSparseEmbedder` — deterministic,
  download-free embedders for tests and examples.
* :func:`embed_many` — batched, semaphore-bounded helper.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
from rag_core.protocols import Embedder, SparseEmbedder
from rag_core.types import SparseVector

__all__ = [
    "FastEmbedDense",
    "FastEmbedSparse",
    "MockEmbedder",
    "MockSparseEmbedder",
    "embed_many",
]

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dense embedders
# --------------------------------------------------------------------------- #


class FastEmbedDense(Embedder):
    """Dense embedder backed by ``fastembed.TextEmbedding``.

    The model is loaded lazily on the first call to :meth:`ensure_model`; the
    constructor itself never downloads weights.  For testing, inject a
    ``model_factory`` callable that returns a fake model object.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        max_threads: int = 1,
        normalize: bool = True,
        model_factory: Callable[[], object] | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_threads = max_threads
        self._normalize = normalize
        self._model_factory = model_factory
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def dim(self) -> int | None:
        """Embedding dimensionality, or ``None`` if not yet resolvable."""
        if self._model is not None:
            d = getattr(self._model, "dim", None)
            if d is not None:
                return int(d)
        # Best-effort static lookup.
        try:
            from fastembed import TextEmbedding

            for entry in TextEmbedding.list_supported_models():
                if entry.get("model") == self._model_name:
                    return entry.get("dim")
        except Exception:
            pass
        return None

    def ensure_model(self) -> object:
        """Load and cache the underlying model on first access."""
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory()
        else:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self._model_name,
                threads=self._max_threads,
                lazy_load=True,
            )
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        model = self.ensure_model()
        text_list = list(texts)
        if not text_list:
            return []
        return list(await asyncio.to_thread(self._embed_sync, text_list, model))

    def _embed_sync(self, texts: list[str], model: object) -> list[list[float]]:
        """Synchronous embedding loop (runs in a worker thread)."""
        batch_size = self._batch_size
        results: list[list[float]] = []
        embed_iter = cast(Any, model).embed(texts, batch_size=batch_size)
        for vec in embed_iter:
            dense = [float(v) for v in np.asarray(vec).ravel()]
            if self._normalize:
                dense = _l2_normalize(dense)
            results.append(dense)
        return results


class MockEmbedder(Embedder):
    """Deterministic, hash-based dense embedder (no downloads).

    Each text maps to a fixed ``dim``-dimensional unit vector derived from the
    SHA-256 hash of the text.
    """

    def __init__(self, dim: int = 384, normalize: bool = True, seed: str = "") -> None:
        self._dim = dim
        self._normalize = normalize
        self._seed = seed

    @property
    def model_name(self) -> str:
        return "mock-dense"

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def normalize(self) -> bool:
        return self._normalize

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return list(await asyncio.to_thread(self._embed_sync, list(texts)))

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            h = hashlib.sha256((self._seed + text).encode("utf-8")).digest()
            vec = [h[i % len(h)] / 255.0 for i in range(self._dim)]
            if self._normalize:
                vec = _l2_normalize(vec)
            results.append(vec)
        return results


# --------------------------------------------------------------------------- #
# Sparse embedders
# --------------------------------------------------------------------------- #


class FastEmbedSparse(SparseEmbedder):
    """Sparse embedder backed by ``fastembed.SparseTextEmbedding``."""

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        batch_size: int = 64,
        max_threads: int = 1,
        model_factory: Callable[[], object] | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._max_threads = max_threads
        self._model_factory = model_factory
        self._model: object | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        if self._model_factory is not None:
            self._model = self._model_factory()
        else:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(
                model_name=self._model_name,
                threads=self._max_threads,
                lazy_load=True,
            )
        return self._model

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        model = self.ensure_model()
        text_list = list(texts)
        if not text_list:
            return []
        return list(await asyncio.to_thread(self._embed_sync, text_list, model))

    def _embed_sync(self, texts: list[str], model: object) -> list[SparseVector]:
        batch_size = self._batch_size
        results: list[SparseVector] = []
        embed_iter = cast(Any, model).embed(texts, batch_size=batch_size)
        for sparse in embed_iter:
            indices = [int(i) for i in sparse.indices]
            values = [float(v) for v in sparse.values]
            # Sort by index for deterministic ordering.
            pairs = sorted(zip(indices, values, strict=True))
            indices = [p[0] for p in pairs]
            values = [p[1] for p in pairs]
            results.append(SparseVector(indices=indices, values=values))
        return results


class MockSparseEmbedder(SparseEmbedder):
    """Deterministic BM25-style sparse embedder (no downloads).

    Each whitespace token is hashed to an index in ``[0, vocab_size)``.  The
    value is the term frequency.
    """

    def __init__(self, vocab_size: int = 30522, seed: str = "") -> None:
        self._vocab_size = vocab_size
        self._seed = seed

    @property
    def model_name(self) -> str:
        return "mock-sparse"

    async def embed_sparse(self, texts: Sequence[str]) -> list[SparseVector]:
        return list(await asyncio.to_thread(self._embed_sync, list(texts)))

    def _embed_sync(self, texts: list[str]) -> list[SparseVector]:
        results: list[SparseVector] = []
        for text in texts:
            counts: dict[int, float] = {}
            for token in _tokenize_text(text):
                idx = (
                    int(hashlib.sha256((self._seed + token).encode("utf-8")).hexdigest(), 16)
                    % self._vocab_size
                )
                counts[idx] = counts.get(idx, 0.0) + 1.0
            indices = sorted(counts.keys())
            values = [counts[i] for i in indices]
            results.append(SparseVector(indices=indices, values=values))
        return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_text(text: str) -> list[str]:
    """Lowercase word-tokenize for the mock sparse embedder."""
    return _TOKEN_RE.findall(text.lower())


async def embed_many(
    texts: Sequence[str],
    embedder: Embedder,
    batch_size: int = 64,
    max_concurrency: int = 4,
) -> list[list[float]]:
    """Embed *texts* in batches, bounding concurrency with a semaphore."""
    if not texts:
        return []
    text_list = list(texts)
    batches = [text_list[i : i + batch_size] for i in range(0, len(text_list), batch_size)]
    sem = asyncio.Semaphore(max_concurrency)

    async def _run(batch: list[str]) -> list[list[float]]:
        async with sem:
            return await embedder.embed(batch)

    results = await asyncio.gather(*[_run(b) for b in batches])
    return [v for batch_result in results for v in batch_result]
