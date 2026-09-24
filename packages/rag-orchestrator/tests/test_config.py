"""Tests for PipelineConfig: validation, local_default, per-request overrides."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_orchestrator.config import PipelineConfig, pipeline_schema_version


def test_schema_version() -> None:
    assert pipeline_schema_version == "1.0"
    assert PipelineConfig().schema_version == pipeline_schema_version


def test_unknown_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(nonexistent_stage={"enable": True})


def test_local_default_profile() -> None:
    cfg = PipelineConfig.local_default()
    assert cfg.name == "local_fast"
    assert cfg.ingestion.strategy == "pymupdf"
    assert cfg.embedding.strategy == "recursive"
    assert cfg.retrieval.strategy == "rrf"
    assert cfg.retrieval.overrides["top_k"] == 10
    assert cfg.rerank.overrides["max_candidates"] == 50
    assert cfg.context.overrides["token_budget"] == 2048
    assert cfg.generation.overrides["base_url"] == "http://localhost:1234/v1"


def test_merge_overrides_routes_known_keys() -> None:
    cfg = PipelineConfig.local_default().merge_overrides(
        {"top_k": 5, "token_budget": 512, "temperature": 0.7, "timeout_s": 3.5}
    )
    assert cfg.retrieval.overrides["top_k"] == 5
    assert cfg.token_budget == 512
    assert cfg.context.overrides["token_budget"] == 512
    assert cfg.generation.overrides["temperature"] == 0.7
    assert cfg.timeout_s == 3.5
    # original config is untouched
    assert PipelineConfig.local_default().retrieval.overrides["top_k"] == 10


def test_merge_overrides_unknown_key_lands_in_generation() -> None:
    cfg = PipelineConfig().merge_overrides({"provider_specific_opt": 1})
    assert cfg.generation.overrides["provider_specific_opt"] == 1


def test_merge_overrides_none_is_noop() -> None:
    cfg = PipelineConfig.local_default()
    assert cfg.merge_overrides(None) is cfg


def test_merge_overrides_max_candidates_routes_to_rerank() -> None:
    cfg = PipelineConfig().merge_overrides({"max_candidates": 20})
    assert cfg.rerank.overrides["max_candidates"] == 20


def test_merge_overrides_cost_budget_and_concurrency() -> None:
    cfg = PipelineConfig().merge_overrides({"cost_budget_usd": 0.50, "concurrency": 8})
    assert cfg.cost_budget_usd == 0.50
    assert cfg.concurrency == 8
