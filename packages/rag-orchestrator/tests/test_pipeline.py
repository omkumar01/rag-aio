"""Tests for the declarative Pipeline spec + registry (YAML round-trip)."""

from __future__ import annotations

import fakes
from rag_orchestrator.config import PipelineConfig
from rag_orchestrator.pipeline import (
    Pipeline,
    PipelineRegistry,
    default_pipeline,
    default_registry,
)


def test_default_pipeline_shape() -> None:
    p = default_pipeline()
    assert p.config.name == "local_fast"
    assert p.to_dict()["schema_version"] == "1.0"
    assert p.to_dict()["config"]["name"] == "local_fast"


def test_yaml_roundtrip(tmp_path) -> None:
    p = Pipeline.from_config(PipelineConfig.local_default(), services=None)
    path = tmp_path / "pipeline.yaml"
    p.save_yaml(str(path))
    loaded = Pipeline.load_yaml(str(path))
    assert loaded.config == p.config
    assert loaded.version == p.version


def test_registry_add_get_list_remove() -> None:
    registry = PipelineRegistry()
    p1 = Pipeline.from_config(PipelineConfig(name="a"))
    p2 = Pipeline.from_config(PipelineConfig(name="b"))
    registry.add(p1)
    registry.add(p2)
    assert registry.get("a") is p1
    assert {p.config.name for p in registry.list()} == {"a", "b"}
    assert registry.remove("a") is True
    assert registry.get("a") is None
    assert registry.remove("a") is False


def test_registry_save_load_roundtrip(tmp_path) -> None:
    registry = default_registry()
    path = tmp_path / "registry.yaml"
    registry.save(str(path))
    loaded = PipelineRegistry.load(str(path))
    assert {p.config.name for p in loaded.list()} == {"local_fast"}


def test_from_config_with_services_records_component_types() -> None:
    """from_config with a services bag records component type names (auditable)."""
    services = fakes.make_services()
    p = Pipeline.from_config(PipelineConfig.local_default(), services=services)
    assert p.components["retriever"] == "StubRetriever"
    assert p.components["reranker"] == "StubReranker"
    assert p.components["generator"] == "StubGenerator"
