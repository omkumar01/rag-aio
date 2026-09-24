"""Tests for RAGConfig: defaults, mock profile, and TOML round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rag_aio.config import (
    EmbedderConfig,
    GenerationConfig,
    RAGConfig,
    StorageConfig,
)

# --------------------------------------------------------------------------- #
# Minimal TOML serializer (stdlib has no TOML writer; tomli_w not installed)
# --------------------------------------------------------------------------- #


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot serialize {type(value)!r} to TOML")


def _dict_to_toml(data: dict, prefix: str = "") -> list[str]:
    """Serialize a *plain dict* (from ``model_dump(mode="json")``) to TOML lines.

    ``None`` values are omitted so Pydantic fills in defaults on read-back.
    Empty dicts are omitted for the same reason.
    """
    lines: list[str] = []
    scalars: list[tuple[str, object]] = []
    tables: list[tuple[str, dict]] = []
    for key, val in data.items():
        if isinstance(val, dict):
            if val:
                tables.append((key, val))
        elif val is not None:
            scalars.append((key, val))

    for key, val in scalars:
        lines.append(f"{key} = {_toml_value(val)}")

    for key, val in tables:
        full_path = f"{prefix}.{key}" if prefix else key
        lines.append("")
        lines.append(f"[{full_path}]")
        lines.extend(_dict_to_toml(val, full_path))

    return lines


def to_toml(data: dict) -> str:
    return "\n".join(_dict_to_toml(data))


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_defaults() -> None:
    cfg = RAGConfig()
    assert cfg.embedder.backend == "fastembed"
    assert cfg.embedder.dim == 384
    assert cfg.storage.qdrant_path == "./data/qdrant"
    assert cfg.storage.db_url == "sqlite+aiosqlite:///./data/rag.db"
    assert cfg.storage.cache_backend == "memory"
    assert cfg.generation.provider == "lm_studio"
    assert cfg.generation.base_url == "http://localhost:1234/v1"
    assert cfg.generation.temperature == 0.2


def test_field_configs() -> None:
    assert EmbedderConfig().backend == "fastembed"
    assert StorageConfig().cache_backend == "memory"
    assert GenerationConfig().provider == "lm_studio"


# --------------------------------------------------------------------------- #
# Mock profile
# --------------------------------------------------------------------------- #


def test_mock_uses_mock_backend() -> None:
    cfg = RAGConfig.mock()
    assert cfg.embedder.backend == "mock"


def test_mock_overrides_embedding_policy() -> None:
    cfg = RAGConfig.mock()
    assert cfg.pipeline.embedding.policy == "mock"


def test_mock_preserves_local_default_name() -> None:
    cfg = RAGConfig.mock()
    assert cfg.pipeline.name == "local_fast"


# --------------------------------------------------------------------------- #
# TOML round-trip
# --------------------------------------------------------------------------- #


def test_mock_config_round_trip(tmp_path: Path) -> None:
    original = RAGConfig.mock()
    toml_str = to_toml(original.model_dump(mode="json"))
    path = tmp_path / "config.toml"
    path.write_text(toml_str, encoding="utf-8")

    loaded = RAGConfig.from_file(path)
    assert loaded == original


def test_from_file_with_minimal_overrides(tmp_path: Path) -> None:
    """A hand-written TOML with only mock overrides reads back correctly."""
    toml_str = """\
[embedder]
backend = "mock"

[pipeline.embedding]
policy = "mock"
strategy = "recursive"
"""
    path = tmp_path / "partial.toml"
    path.write_text(toml_str, encoding="utf-8")

    cfg = RAGConfig.from_file(path)
    assert cfg.embedder.backend == "mock"
    assert cfg.pipeline.embedding.policy == "mock"
    assert cfg.pipeline.embedding.strategy == "recursive"
    assert cfg.storage.cache_backend == "memory"


def test_from_file_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError):
        RAGConfig.from_file(missing)


def test_secret_by_reference() -> None:
    """api_key_ref is an env-var name, not a credential."""
    cfg = RAGConfig(generation=GenerationConfig(api_key_ref="MY_API_KEY"))
    dumped = cfg.model_dump(mode="json")
    assert dumped["generation"]["api_key_ref"] == "MY_API_KEY"
