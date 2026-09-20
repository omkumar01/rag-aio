"""Shared test fixtures for rag-context."""

from __future__ import annotations

from typing import Any

import pytest
from rag_context.tokenizer import WhitespaceCounter
from rag_core.queries import Query
from rag_core.retrieval import RetrievalHit
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


@pytest.fixture
def whitespace_tokenizer() -> WhitespaceCounter:
    return WhitespaceCounter()


@pytest.fixture
def wordlevel_tokenizer() -> Tokenizer:
    vocab = {
        "[UNK]": 0,
        "hello": 1,
        "world": 2,
        "foo": 3,
        "bar": 4,
        "baz": 5,
        "rag": 6,
        "aio": 7,
        "context": 8,
        "builder": 9,
        "alpha": 10,
        "beta": 11,
        "gamma": 12,
        "delta": 13,
        "epsilon": 14,
        "zeta": 15,
        "neighbor": 16,
        "one": 17,
        "two": 18,
        "three": 19,
        "four": 20,
        "five": 21,
        "six": 22,
        "seven": 23,
        "eight": 24,
        "nine": 25,
        "ten": 26,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


@pytest.fixture
def simple_query() -> Query:
    return Query(text="what is rag-aio?")


@pytest.fixture
def make_hit() -> Any:
    """Factory building minimal :class:`RetrievalHit` instances for tests."""

    def _make(
        chunk_id: str,
        document_id: str,
        score: float,
        text: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
        strategy: str = "dense",
        **extra: Any,
    ) -> RetrievalHit:
        base: dict[str, Any] = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "score": score,
            "normalized_score": score,
            "rank": 0,
            "strategy": strategy,
        }
        if text is not None:
            base["text"] = text
        if metadata:
            base["metadata"] = metadata
        base.update(extra)
        return RetrievalHit(**base)

    return _make
