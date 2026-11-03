"""Tests for lifecycle helpers: closing sync/async components best-effort."""

from __future__ import annotations

import pytest
from rag_core import Closable, aclose_component
from rag_core.lifecycle import Closable as ClosableFromModule
from rag_core.lifecycle import aclose_component as aclose_component_from_module


@pytest.mark.unit
def test_closable_exported_and_runtime_checkable() -> None:
    assert Closable is ClosableFromModule
    assert aclose_component is aclose_component_from_module

    class WithClose:
        def close(self) -> None: ...

    class WithoutClose:
        pass

    assert isinstance(WithClose(), Closable)
    assert not isinstance(WithoutClose(), Closable)


@pytest.mark.unit
async def test_aclose_component_async_only() -> None:
    closed: list[str] = []

    class AsyncOnly:
        async def aclose(self) -> None:
            closed.append("aclose")

    obj = AsyncOnly()
    assert await aclose_component(obj) is True
    assert closed == ["aclose"]


@pytest.mark.unit
async def test_aclose_component_sync_only() -> None:
    closed: list[str] = []

    class SyncOnly:
        def close(self) -> None:
            closed.append("close")

    obj = SyncOnly()
    assert await aclose_component(obj) is True
    assert closed == ["close"]


@pytest.mark.unit
async def test_aclose_component_sync_close_returning_awaitable() -> None:
    closed: list[str] = []

    async def _awaitable() -> None:
        closed.append("awaited")

    class SyncReturningAwaitable:
        def close(self) -> object:
            return _awaitable()

    obj = SyncReturningAwaitable()
    assert await aclose_component(obj) is True
    assert closed == ["awaited"]


@pytest.mark.unit
async def test_aclose_component_neither_method_returns_false() -> None:
    class Plain:
        pass

    assert await aclose_component(Plain()) is False


@pytest.mark.unit
async def test_aclose_component_prefers_aclose() -> None:
    closed: list[str] = []

    class Both:
        async def aclose(self) -> None:
            closed.append("aclose")

        def close(self) -> None:
            closed.append("close")

    obj = Both()
    assert await aclose_component(obj) is True
    assert closed == ["aclose"]
