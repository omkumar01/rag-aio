"""Local (in-process) rerankers: CrossEncoder and a zero-dependency heuristic."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol, cast

from rag_core import Query, RerankError, RetrievalHit

from .base import Scorer, rerank_candidates
from .config import RerankConfig

__all__ = ["CrossEncoderReranker", "HeuristicReranker"]

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_TERM_RE = re.compile(r"[a-z0-9]+")


class _CrossEncoderLike(Protocol):
    """Structural type for sentence-transformers CrossEncoder (and test fakes)."""

    def predict(self, pairs: list[list[str]]) -> list[float]: ...


class CrossEncoderReranker:
    """Reranker backed by a sentence-transformers ``CrossEncoder``.

    The model is loaded lazily on first use. Tests may inject
    ``model_factory`` (returning an object with ``predict(pairs)``);
    otherwise the optional ``sentence-transformers`` extra is imported. If it
    is missing and no factory is supplied, a helpful :class:`RerankError` is
    raised. Inference runs in a worker thread via :func:`asyncio.to_thread`.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        config: RerankConfig | None = None,
        model_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._config = config or RerankConfig()
        self._model_factory = model_factory
        self._model: _CrossEncoderLike | None = None
        self._lock = asyncio.Lock()

    async def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if not candidates:
            return []
        model = await self._get_model()
        scorer = self._make_scorer(model)
        cfg = self._config.model_copy(update={"top_k": top_k})
        hits = await rerank_candidates(query.text, list(candidates), scorer, cfg)
        for hit in hits:
            hit.model = self._model_name
        return hits

    async def _get_model(self) -> _CrossEncoderLike:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await self._load_model()
        return self._model

    async def _load_model(self) -> _CrossEncoderLike:
        if self._model_factory is not None:
            loaded = await asyncio.to_thread(self._model_factory)
            return cast(_CrossEncoderLike, loaded)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankError(
                "sentence_transformers is not installed; install with "
                "`uv add 'rag-rerank[sentence-transformers]'` or pass a "
                "model_factory"
            ) from exc
        loaded = await asyncio.to_thread(CrossEncoder, self._model_name)
        return cast(_CrossEncoderLike, loaded)

    def _make_scorer(self, model: _CrossEncoderLike) -> Scorer:
        async def scorer(query: str, texts: list[str]) -> list[float]:
            pairs = [[query, text] for text in texts]
            result = await asyncio.to_thread(model.predict, pairs)
            return list(result)

        return scorer


class HeuristicReranker:
    """Zero-dependency fallback reranker based on term-overlap (Jaccard).

    Scores each candidate by the Jaccard similarity of the query's term set
    against the candidate text's term set. Useful as a documented baseline
    and for deterministic tests. When ``keyword_overlap`` is ``False`` the
    scorer returns 0.0 for every candidate (no signal).
    """

    def __init__(
        self,
        config: RerankConfig | None = None,
        keyword_overlap: bool = True,
    ) -> None:
        self._config = config or RerankConfig()
        self._keyword_overlap = keyword_overlap

    async def rerank(
        self,
        query: Query,
        candidates: Sequence[RetrievalHit],
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        if not candidates:
            return []
        scorer = self._make_scorer()
        cfg = self._config.model_copy(update={"top_k": top_k})
        hits = await rerank_candidates(query.text, list(candidates), scorer, cfg)
        for hit in hits:
            hit.model = "heuristic"
        return hits

    def _make_scorer(self) -> Scorer:
        if not self._keyword_overlap:

            async def zero(query: str, texts: list[str]) -> list[float]:
                return [0.0 for _ in texts]

            return zero

        async def scorer(query: str, texts: list[str]) -> list[float]:
            qterms = set(_TERM_RE.findall(query.lower()))
            out: list[float] = []
            for text in texts:
                terms = set(_TERM_RE.findall(text.lower())) if text else set()
                if not qterms or not terms:
                    out.append(0.0)
                else:
                    out.append(len(qterms & terms) / len(qterms | terms))
            return out

        return scorer
