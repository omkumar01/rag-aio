"""Shared fixtures and helpers for rag-generation tests (all offline)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
from rag_core.generation import GenerationRequest, Message


def _make_request(**overrides: Any) -> GenerationRequest:
    defaults: dict[str, Any] = {
        "messages": [Message(role="user", content="Hello")],
        "model": "test-model",
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


@pytest.fixture
def make_request() -> Callable[..., GenerationRequest]:
    """Build a :class:`GenerationRequest` with sensible defaults."""
    return _make_request


def _wire(
    provider_cls: Any,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "test-key",
    **kwargs: Any,
) -> Any:
    """Instantiate ``provider_cls`` backed by a :class:`httpx.MockTransport`."""
    transport = httpx.MockTransport(handler)
    return provider_cls(api_key_provider=lambda: api_key, transport=transport, **kwargs)


@pytest.fixture
def wire() -> Callable[..., Any]:
    """Factory wiring a provider class to a MockTransport handler."""
    return _wire
