"""In-memory store implementations (numpy-free, pure Python).

Provides :class:`InMemoryVectorStore` (brute-force cosine similarity) and
:class:`InMemoryKeyValueStore`. Together with the in-process SQL sqlite stores
these let the full persistence layer be exercised without any external service.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from rag_core.protocols import KeyValueStore, VectorStore
from rag_core.retrieval import RetrievalHit


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors (numpy-free)."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    norm = math.sqrt(na) * math.sqrt(nb)
    if norm == 0.0:
        return 0.0
    return dot / norm


def _normalize_score(score: float) -> float:
    """Map a cosine similarity in [-1, 1] to a normalized score in [0, 1]."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _matches(point_payload: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Equality filter check; ``metadata.<k>`` keys traverse nested metadata."""
    if not filters:
        return True
    for key, value in filters.items():
        if key.startswith("metadata."):
            meta = point_payload.get("metadata") or {}
            if not isinstance(meta, dict) or meta.get(key[len("metadata.") :]) != value:
                return False
        else:
            if point_payload.get(key) != value:
                return False
    return True


class InMemoryVectorStore(VectorStore):
    """Brute-force in-memory :class:`VectorStore` using cosine similarity.

    Parameters
    ----------
    vector_size:
        Expected dimensionality (used for assertions only; not enforced at
        runtime to allow flexibility in tests).
    """

    def __init__(self, vector_size: int = 384) -> None:
        self._vector_size = vector_size
        self._points: list[dict[str, Any]] = []

    async def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict[str, Any],
        namespace: str | None = None,
    ) -> None:
        ns = namespace or "default"
        for point in self._points:
            if point["chunk_id"] == chunk_id and point["namespace"] == ns:
                point["vector"] = [float(v) for v in vector]
                point["payload"] = dict(payload)
                point["namespace"] = ns
                return
        self._points.append(
            {
                "chunk_id": chunk_id,
                "vector": [float(v) for v in vector],
                "payload": dict(payload),
                "namespace": ns,
            }
        )

    async def upsert_many(
        self,
        items: list[tuple[str, list[float], dict[str, Any]]],
        namespace: str | None = None,
    ) -> None:
        for chunk_id, vector, payload in items:
            await self.upsert(chunk_id, vector, payload, namespace)

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalHit]:
        ns = namespace or "default"
        scored: list[tuple[float, dict[str, Any]]] = []
        for point in self._points:
            if point["namespace"] != ns:
                continue
            if not _matches(point["payload"], filters):
                continue
            score = _cosine([float(v) for v in vector], point["vector"])
            scored.append((score, point))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[RetrievalHit] = []
        for rank, (score, point) in enumerate(scored[: max(top_k, 0)]):
            payload = point["payload"]
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            hits.append(
                RetrievalHit(
                    chunk_id=payload.get("chunk_id", point["chunk_id"]),
                    document_id=payload.get("document_id", ""),
                    score=float(score),
                    normalized_score=_normalize_score(float(score)),
                    rank=rank,
                    strategy="dense",
                    text=payload.get("text"),
                    metadata=metadata,
                    filters_applied=dict(filters or {}),
                )
            )
        return hits

    async def delete(
        self,
        chunk_ids: Sequence[str],
        namespace: str | None = None,
    ) -> None:
        ns = namespace or "default"
        wanted = set(chunk_ids)
        self._points = [
            p for p in self._points if not (p["namespace"] == ns and p["chunk_id"] in wanted)
        ]

    def count(self, namespace: str | None = None) -> int:
        """Count stored points, optionally scoped to a namespace (synchronous)."""
        ns = namespace or "default"
        return sum(1 for p in self._points if p["namespace"] == ns)

    async def health(self) -> bool:
        return True


class InMemoryKeyValueStore(KeyValueStore):
    """Dict-backed :class:`KeyValueStore` scoped to a fixed namespace."""

    def __init__(self, namespace: str = "default") -> None:
        self._namespace = namespace
        self._data: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self._data.get(key)

    async def set(self, key: str, value: bytes) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def health(self) -> bool:
        return True


__all__ = ["InMemoryKeyValueStore", "InMemoryVectorStore"]
