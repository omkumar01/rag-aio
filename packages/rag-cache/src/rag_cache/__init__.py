"""rag-cache: multi-level caching for rag-aio stages.

Public API is intentionally small: a key constructor, a typed configuration
object + factory, three async backends (memory / disk / sqlite), a namespace
decorator with ``get_or_set``, and a statistics wrapper. Redis is *not*
implemented here -- it is a documented extension point behind the optional
``redis`` extra.
"""

from __future__ import annotations

from .backends.disk import DiskCache
from .backends.memory import MemoryCache
from .backends.sqlite import SQLiteCache
from .config import CacheConfig, create_cache
from .keys import cache_key
from .namespaced import NamespacedCache
from .stats import CacheStats, InstrumentedCache

__all__ = [
    "CacheConfig",
    "CacheStats",
    "DiskCache",
    "InstrumentedCache",
    "MemoryCache",
    "NamespacedCache",
    "SQLiteCache",
    "cache_key",
    "create_cache",
]

__version__ = "0.1.0"
