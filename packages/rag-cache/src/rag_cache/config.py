"""Typed configuration and the backend factory for rag-cache."""

from __future__ import annotations

from typing import Literal

from rag_core.base import RagBaseModel
from rag_core.errors import ConfigError
from rag_core.protocols import Cache

from .backends.disk import DiskCache
from .backends.memory import MemoryCache
from .backends.sqlite import SQLiteCache


class CacheConfig(RagBaseModel):
    """Configuration for a cache backend.

    ``backend`` selects the implementation; the remaining fields configure it.
    ``namespace`` is a logical grouping used with :func:`rag_cache.keys.cache_key`
    / :class:`rag_cache.namespaced.NamespacedCache` -- it is intentionally not
    consumed by :func:`create_cache` itself so the raw backend stays a pure
    ``Cache``. ``max_bytes`` is enforced by the memory and disk backends (the
    latter best-effort by mtime); SQLite has no in-process byte limit.
    """

    backend: Literal["memory", "disk", "sqlite"] = "memory"
    namespace: str = "default"
    max_items: int = 10_000
    max_bytes: int | None = None
    directory: str | None = None
    path: str | None = None
    default_ttl: float | None = None


def create_cache(config: CacheConfig) -> Cache:
    """Build a :class:`Cache` from a :class:`CacheConfig`.

    Validates backend-specific required fields and raises
    :class:`rag_core.errors.ConfigError` when they are missing.
    """
    if config.backend == "memory":
        return MemoryCache(
            max_items=config.max_items,
            max_bytes=config.max_bytes,
            default_ttl=config.default_ttl,
        )
    if config.backend == "disk":
        if not config.directory:
            raise ConfigError("disk backend requires a 'directory' path")
        return DiskCache(
            directory=config.directory,
            max_bytes=config.max_bytes,
            default_ttl=config.default_ttl,
        )
    if config.backend == "sqlite":
        if not config.path:
            raise ConfigError("sqlite backend requires a 'path' database file")
        return SQLiteCache(path=config.path, default_ttl=config.default_ttl)
    raise ConfigError(f"unknown backend: {config.backend!r}")
