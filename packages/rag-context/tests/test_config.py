"""Tests for :mod:`rag_context.config`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_context import ContextConfig


def test_defaults() -> None:
    cfg = ContextConfig()
    assert cfg.strategy == "relevance_first"
    assert cfg.token_budget == 2048
    assert cfg.reserve_for_answer == 0
    assert cfg.max_per_document == 0
    assert cfg.dedup is True
    assert cfg.merge_overlapping is True
    assert cfg.expand_neighbors is False
    assert cfg.neighbor_window == 1
    assert cfg.min_score is None
    assert cfg.citation_style == "numeric"
    assert cfg.max_item_chars == 8000


def test_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ContextConfig(strategy="relevance_first", bogus=1)  # type: ignore[call-arg]


def test_unknown_strategy_rejected() -> None:
    with pytest.raises(ValidationError):
        ContextConfig(strategy="nope")  # type: ignore[arg-type]
