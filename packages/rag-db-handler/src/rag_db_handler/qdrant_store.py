"""Qdrant vector store adapter (local embedded mode by default, server mode via config).

Implements the :class:`rag_core.protocols.VectorStore` protocol. Local mode uses
``qdrant_client.AsyncQdrantClient(path=...)`` (persistent, single-process as per
ADR-0003); server mode uses ``AsyncQdrantClient(url=..., api_key=...)``.

Qdrant requires UUID or integer point ids. We derive a deterministic UUIDv5 from
``f"{namespace}:{chunk_id}"`` so re-upserting the same chunk is idempotent, and
store the original ``chunk_id`` in the payload for retrieval reconstruction.
"""

# ``qdrant-client`` is an optional backend (the ``qdrant`` extra), so it is
# resolved lazily on first store construction rather than at module import.
# It ships a ``py.typed`` marker, so mypy would normally analyze its type
# stubs; qdrant-client imports ``numpy`` at module load, and the installed
# numpy stubs use PEP 742 ``type`` syntax this project's mypy target cannot
# parse -- a fatal environment error. The repository mypy config already
# intends qdrant_client to be treated as untyped (``ignore_missing_imports``);
# to honor that intent regardless of the shipped py.typed we load the client
# through ``importlib`` and keep only ``Any``-typed aliases, so mypy never
# follows into qdrant_client/numpy.
from __future__ import annotations

import importlib
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from rag_core.errors import ConfigError
from rag_core.protocols import VectorStore
from rag_core.retrieval import RetrievalHit

if TYPE_CHECKING:
    from .config import VectorStoreConfig

    # Alias declarations only (never imported at runtime — see _load_qdrant);
    # kept Any-typed so mypy never follows into qdrant_client/numpy stubs.
    AsyncQdrantClient = Any
    Distance = Any
    FieldCondition = Any
    Filter = Any
    MatchAny = Any
    MatchValue = Any
    PointIdsList = Any
    PointStruct = Any
    VectorParams = Any

_QDRANT_NAMES = (
    "AsyncQdrantClient",
    "Distance",
    "FieldCondition",
    "Filter",
    "MatchAny",
    "MatchValue",
    "PointIdsList",
    "PointStruct",
    "VectorParams",
)

_qdrant: Any = None
_models: Any = None


def _load_qdrant() -> None:
    """Import ``qdrant_client`` on first use and bind the aliases below."""
    global _qdrant, _models
    if _qdrant is None:
        try:
            _qdrant = importlib.import_module("qdrant_client")
            _models = importlib.import_module("qdrant_client.http.models")
        except ImportError as exc:
            msg = (
                "rag_db_handler.qdrant_store needs the 'qdrant-client' package. "
                'Install it with: pip install "rag-aio[qdrant]".'
            )
            raise ImportError(msg) from exc
        for name in _QDRANT_NAMES:
            source = _qdrant if name == "AsyncQdrantClient" else _models
            globals()[name] = getattr(source, name)


def __getattr__(name: str) -> Any:
    """Serve the qdrant aliases to ``from ... import`` users lazily."""
    if name in _QDRANT_NAMES:
        _load_qdrant()
        return globals()[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


_DISTANCE_MAP: dict[str, Any] | None = None

_UUID_NAMESPACE = uuid.NAMESPACE_DNS


def _distance_map() -> dict[str, Any]:
    global _DISTANCE_MAP
    if _DISTANCE_MAP is None:
        _load_qdrant()
        distance = globals()["Distance"]
        _DISTANCE_MAP = {
            "cosine": distance.COSINE,
            "euclid": distance.EUCLID,
            "dot": distance.DOT,
        }
    return _DISTANCE_MAP


def _point_id(namespace: str | None, chunk_id: str) -> str:
    """Deterministic UUIDv5 point id derived from namespace + chunk_id."""
    return str(uuid.uuid5(_UUID_NAMESPACE, f"{namespace or 'default'}:{chunk_id}"))


def _normalize_score(score: float) -> float:
    """Map a cosine similarity in [-1, 1] to a normalized score in [0, 1]."""
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _build_filter(filters: dict[str, Any] | None, namespace: str | None) -> Filter | None:
    """Build a Qdrant ``Filter`` from equality predicates.

    Scalar values produce ``MatchValue`` equality conditions. Keys prefixed with
    ``metadata.`` map to Qdrant dotted field keys which traverse the nested
    ``metadata`` payload object. List values produce ``MatchAny``.
    """
    must: list[FieldCondition] = []
    if namespace is not None:
        must.append(
            _models.FieldCondition(key="namespace", match=_models.MatchValue(value=namespace))
        )
    for key, value in (filters or {}).items():
        if value is None:
            continue
        if isinstance(value, bool):
            must.append(_models.FieldCondition(key=key, match=_models.MatchValue(value=value)))
        elif isinstance(value, (list, tuple)):
            must.append(_models.MatchAny(any=list(value)))
        else:
            must.append(_models.FieldCondition(key=key, match=_models.MatchValue(value=value)))
    if not must:
        return None
    return _models.Filter(must=must)


class QdrantVectorStore(VectorStore):
    """Vector store backed by Qdrant (async client).

    Parameters
    ----------
    config:
        A :class:`VectorStoreConfig`. Local mode (default) persists to
        ``config.path``; server mode connects to ``config.url``. The
        ``api_key_ref`` (if any) is resolved from the environment here and
        forwarded to the client only -- it is never stored on the instance.
    """

    def __init__(self, config: VectorStoreConfig) -> None:
        _load_qdrant()
        config.assert_valid()
        if config.backend != "qdrant":
            raise ConfigError(f"unsupported vector store backend: {config.backend}")

        self._config = config
        self._collection: str = config.collection
        self._vector_size: int = config.vector_size
        self._distance: Any = _distance_map()[config.distance]
        self._ready: bool = False

        api_key: str | None = None
        if config.mode == "server" and config.api_key_ref is not None:
            from .config import resolve_env_secret

            api_key = resolve_env_secret(config.api_key_ref)

        if config.mode == "local":
            self._client = _qdrant.AsyncQdrantClient(path=config.path, prefer_grpc=False)
        else:
            self._client = _qdrant.AsyncQdrantClient(
                url=config.url, api_key=api_key, prefer_grpc=False
            )

    def __repr__(self) -> str:
        # Deliberately omits any secret material; only exposes non-sensitive config.
        return f"QdrantVectorStore(collection={self._collection!r}, mode={self._config.mode!r})"

    @property
    def collection(self) -> str:
        return self._collection

    async def ensure_collection(self) -> None:
        """Create the collection with configured vector params if it does not exist.

        Idempotent and lazily invoked from upsert/search/count/delete. The
        existence check is cached after the first success.
        """
        if self._ready:
            return
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=_models.VectorParams(
                    size=self._vector_size,
                    distance=self._distance,
                ),
            )
        self._ready = True

    async def upsert(
        self,
        chunk_id: str,
        vector: list[float],
        payload: dict[str, Any],
        namespace: str | None = None,
    ) -> None:
        await self.ensure_collection()
        ns = namespace or "default"
        stored = self._prepare_payload(payload, chunk_id, ns)
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                _models.PointStruct(
                    id=_point_id(ns, chunk_id),
                    vector=[float(v) for v in vector],
                    payload=stored,
                )
            ],
            wait=True,
        )

    async def upsert_many(
        self,
        items: list[tuple[str, list[float], dict[str, Any]]],
        namespace: str | None = None,
    ) -> None:
        """Batch upsert of ``(chunk_id, vector, payload)`` tuples (same namespace)."""
        await self.ensure_collection()
        ns = namespace or "default"
        points: list[Any] = []
        for chunk_id, vector, payload in items:
            points.append(
                _models.PointStruct(
                    id=_point_id(ns, chunk_id),
                    vector=[float(v) for v in vector],
                    payload=self._prepare_payload(payload, chunk_id, ns),
                )
            )
        if points:
            await self._client.upsert(collection_name=self._collection, points=points, wait=True)

    async def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        await self.ensure_collection()
        query_filter = _build_filter(filters, namespace)
        response = await self._client.query_points(
            collection_name=self._collection,
            query=[float(v) for v in vector],
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False,
            limit=max(top_k, 1),
        )
        hits: list[RetrievalHit] = []
        for rank, point in enumerate(response.points):
            payload: dict[str, Any] = point.payload or {}
            metadata = payload.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
            score = float(point.score)
            hits.append(
                RetrievalHit(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    document_id=payload.get("document_id", ""),
                    score=score,
                    normalized_score=_normalize_score(score),
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
        await self.ensure_collection()
        ns = namespace or "default"
        point_ids = [_point_id(ns, cid) for cid in chunk_ids]
        if point_ids:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=_models.PointIdsList(points=point_ids),
                wait=True,
            )

    async def count(self, namespace: str | None = None) -> int:
        """Count points, optionally scoped to a namespace."""
        await self.ensure_collection()
        if namespace is not None:
            count_filter = Filter(
                must=[FieldCondition(key="namespace", match=_models.MatchValue(value=namespace))]
            )
            result = await self._client.count(
                collection_name=self._collection, count_filter=count_filter
            )
        else:
            result = await self._client.count(collection_name=self._collection)
        return int(result.count)

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.close()

    @staticmethod
    def _prepare_payload(payload: dict[str, Any], chunk_id: str, namespace: str) -> dict[str, Any]:
        stored = dict(payload)
        stored["chunk_id"] = chunk_id
        stored["namespace"] = namespace
        return stored


__all__ = ["QdrantVectorStore"]
