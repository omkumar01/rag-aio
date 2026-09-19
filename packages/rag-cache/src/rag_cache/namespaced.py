"""Namespace-prefixed cache with a ``get_or_set`` helper."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TypeGuard, cast

from rag_core.protocols import Cache

from .keys import cache_key

#: A value factory: a plain callable returning ``bytes`` or one returning an
#: awaitable of ``bytes`` (an ``async def`` factory). Both are accepted.
Factory = Callable[[], bytes | Awaitable[bytes]]


def _is_awaitable(obj: object) -> TypeGuard[Awaitable[bytes]]:
    """``True`` when ``obj`` is an awaitable (coroutine/future) of bytes."""
    return inspect.isawaitable(obj)


class NamespacedCache:
    """Decorator that prefixes every key with a namespace.

    This lets a single backend be shared across stages/namespaces without the
    caller remembering to prefix keys. ``get_or_set`` resolves the key, returns
    the cached value when present, and otherwise invokes ``factory`` (sync or
    ``async``) -- storing and returning only the successful result. Exceptions
    from the factory propagate untouched, leaving the cache unchanged.

    Note: ``clear``/``size`` are intentionally *not* exposed: a namespace
    should not wipe or size the whole shared backend. Callers that need those
    on the underlying store operate on ``inner`` directly.

    Args:
        inner: The wrapped cache.
        namespace: Prefix prepended to every key.
        key_hash: When ``True`` (default) each key becomes
            ``cache_key(namespace, key=<key>)`` -- a fixed-length hex digest.
            When ``False`` the literal ``"namespace:<key>"`` string is used.
    """

    def __init__(self, inner: Cache, namespace: str, key_hash: bool = True) -> None:
        self._inner = inner
        self._namespace = namespace
        self._key_hash = key_hash

    def _resolve(self, key: str) -> str:
        if self._key_hash:
            return cache_key(self._namespace, key=key)
        return f"{self._namespace}:{key}"

    async def get(self, key: str) -> bytes | None:
        return await self._inner.get(self._resolve(key))

    async def set(self, key: str, value: bytes, ttl: float | None = None) -> None:
        await self._inner.set(self._resolve(key), value, ttl)

    async def delete(self, key: str) -> None:
        await self._inner.delete(self._resolve(key))

    async def get_or_set(self, key: str, factory: Factory) -> bytes:
        """Return the cached value for ``key``, computing it via ``factory`` on a miss."""
        resolved = self._resolve(key)
        cached = await self._inner.get(resolved)
        if cached is not None:
            return cached
        product: bytes | Awaitable[bytes] = factory()
        if _is_awaitable(product):
            value = await product
        else:
            value = cast(bytes, product)
        await self._inner.set(resolved, value)
        return value
