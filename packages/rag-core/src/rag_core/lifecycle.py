"""Lifecycle helpers for closing components uniformly (sync or async).

Store adapters, clients, and engines across the workspace expose either an
``async aclose()`` (async resources) or a ``close()`` (sync or async-returning).
These helpers let shutdown paths (orchestrator teardown, ``check_health``
callers, examples) release resources without knowing which convention a
component follows.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Closable(Protocol):
    """Component exposing a close/aclose method (sync or async)."""

    def close(self) -> Any: ...


async def aclose_component(obj: Any) -> bool:
    """Best-effort close of ``obj``, awaiting async results.

    Looks for an ``aclose`` method first, then ``close``; calls it with no
    arguments and awaits the result if it is awaitable.

    Returns:
        True if a method was found and called, False if ``obj`` exposes
        neither method. Never raises for a missing method.
    """
    method = getattr(obj, "aclose", None) or getattr(obj, "close", None)
    if not callable(method):
        return False
    result = method()
    if inspect.isawaitable(result):
        await result
    return True


__all__ = ["Closable", "aclose_component"]
