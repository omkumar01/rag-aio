"""Namespace-scoped views over a single underlying vector store.

The web example keeps **one** physical vector store and isolates documents by
namespace: ``user_{id}`` for per-user uploads and ``global`` for admin-managed
shared documents. :class:`CompositeVectorStore` is the read/write window one
user (or the global pool) sees: writes always land in the primary namespace,
searches fan out over every visible namespace and merge the results.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Sequence
from typing import Any

from rag_core.protocols import VectorStore
from rag_core.retrieval import RetrievalHit

__all__ = ["CompositeVectorStore"]


class CompositeVectorStore:
    """A :class:`~rag_core.protocols.VectorStore` view over one base store.

    Args:
        base: The underlying store holding every namespace.
        primary_ns: Namespace all writes (upsert/delete) are routed to.
        search_ns: Namespaces searched on read; hits are merged by score
            (stable ``chunk_id`` tie-break), deduplicated, truncated to
            ``top_k`` and tagged with ``metadata["pool"]``.
    """

    def __init__(self, base: VectorStore, primary_ns: str, search_ns: tuple[str, ...]) -> None:
        self._base = base
        self._primary_ns = primary_ns
        self._search_ns = tuple(search_ns) or (primary_ns,)

    @property
    def base(self) -> VectorStore:
        """The underlying shared store (useful for admin/cross-namespace checks)."""
        return self._base

    @property
    def primary_ns(self) -> str:
        """Namespace every write lands in."""
        return self._primary_ns

    @property
    def search_ns(self) -> tuple[str, ...]:
        """Namespaces visible to reads."""
        return self._search_ns

    async def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict[str, Any],
        namespace: str | None = None,
    ) -> None:
        """Write to the primary namespace (the ``namespace`` arg is overridden)."""
        await self._base.upsert(chunk_id, vector, payload, namespace=self._primary_ns)

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalHit]:
        """Fan out over all visible namespaces and merge the results."""
        if top_k <= 0:
            return []
        per_ns = await asyncio.gather(
            *(
                self._base.search(vector, top_k, filters=filters, namespace=ns)
                for ns in self._search_ns
            )
        )
        best: dict[str, RetrievalHit] = {}
        for ns, hits in zip(self._search_ns, per_ns, strict=True):
            for hit in hits:
                tagged = hit.model_copy(update={"metadata": {**hit.metadata, "pool": ns}})
                current = best.get(tagged.chunk_id)
                if current is None or (-tagged.score, tagged.chunk_id) < (
                    -current.score,
                    current.chunk_id,
                ):
                    best[tagged.chunk_id] = tagged
        merged = sorted(best.values(), key=lambda hit: (-hit.score, hit.chunk_id))
        return [hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(merged[:top_k])]

    async def delete(
        self,
        chunk_ids: Sequence[str],
        namespace: str | None = None,
    ) -> None:
        """Delete from the primary namespace (the ``namespace`` arg is overridden)."""
        await self._base.delete(chunk_ids, namespace=self._primary_ns)

    async def health(self) -> bool:
        return await self._base.health()

    async def count(self, namespace: str | None = None) -> int:
        """Sum per-namespace point counts; ``0`` when the base store cannot count."""
        counter = getattr(self._base, "count", None)
        if not callable(counter):
            return 0
        targets = (namespace,) if namespace is not None else self._search_ns
        total = 0
        for ns in targets:
            outcome = counter(namespace=ns)
            if inspect.isawaitable(outcome):
                total += int(await outcome)
            else:
                total += int(outcome)
        return total
