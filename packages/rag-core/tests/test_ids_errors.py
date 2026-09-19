"""Tests for stable IDs, content hashing, and error taxonomy."""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from rag_core.errors import (
    ConfigError,
    EmbeddingError,
    GenerationError,
    IngestionError,
    ProviderError,
    RagError,
    RetrievalError,
    StorageError,
)
from rag_core.ids import config_hash, content_hash, new_id, stable_id


def test_new_id_unique() -> None:
    assert new_id() != new_id()


def test_new_id_format() -> None:
    value = new_id()
    assert len(value) == 32
    int(value, 16)  # hex


def test_content_hash_stable() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hellö")
    assert content_hash(b"bytes") == content_hash(b"bytes")


def test_content_hash_format() -> None:
    value = content_hash("x")
    assert len(value) == 64
    int(value, 16)


def test_stable_id_deterministic() -> None:
    assert stable_id("doc1", "chunk", 3, "text") == stable_id("doc1", "chunk", 3, "text")
    assert stable_id("doc1", "chunk", 3, "text") != stable_id("doc1", "chunk", 4, "text")


def test_config_hash_ignores_key_order_and_secret_values() -> None:
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_works_on_pydantic_models() -> None:
    class Model(BaseModel):
        x: int
        y: str = "z"

    assert config_hash(Model(x=1)) == config_hash({"x": 1, "y": "z"})


@pytest.mark.parametrize(
    "exc_type",
    [ConfigError, IngestionError, EmbeddingError, RetrievalError, StorageError, GenerationError],
)
def test_error_hierarchy(exc_type: type[RagError]) -> None:
    err = exc_type("something broke", code="X1", details={"k": "v"})
    assert isinstance(err, RagError)
    assert err.code == "X1"
    assert err.details == {"k": "v"}
    assert "something broke" in str(err)


def test_provider_error_subclasses() -> None:
    from rag_core.errors import ProviderUnavailableError, RateLimitError

    assert issubclass(ProviderUnavailableError, ProviderError)
    assert issubclass(RateLimitError, ProviderError)
