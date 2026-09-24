"""Tests for RAGConfig serialization and the config_view helpers.

These keep the Streamlit runtime out of the import graph: they exercise the
Pydantic models shared with the backend plus the headless helpers (masking,
TOML serialization) that live in :mod:`rag_ui.config_view`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rag_aio.config import GenerationConfig, RAGConfig
from rag_ui.config_view import dict_to_toml, mask_secret, validate_save_path

# --------------------------------------------------------------------------- #
# Defaults & profiles
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
    assert cfg.generation.api_key_ref is None
    assert cfg.pipeline.concurrency == 4
    assert cfg.pipeline.cache_reads is True
    assert cfg.pipeline.cache_writes is True


def test_mock_profile_defaults() -> None:
    cfg = RAGConfig.mock()
    assert cfg.embedder.backend == "mock"
    assert cfg.pipeline.embedding.policy == "mock"
    assert cfg.pipeline.name == "local_fast"
    assert cfg.generation.api_key_ref is None


# --------------------------------------------------------------------------- #
# Serialization round-trips
# --------------------------------------------------------------------------- #


def test_round_trip_pydantic() -> None:
    original = RAGConfig()
    assert RAGConfig.model_validate(original.model_dump()) == original


def test_round_trip_json_mode() -> None:
    original = RAGConfig.mock()
    assert RAGConfig.model_validate(original.model_dump(mode="json")) == original


def test_round_trip_toml(tmp_path: Path) -> None:
    original = RAGConfig()
    path = tmp_path / "config.toml"
    path.write_text(dict_to_toml(original.model_dump(mode="json")), encoding="utf-8")
    assert RAGConfig.from_file(path) == original


def test_edited_config_round_trips(tmp_path: Path) -> None:
    data = RAGConfig().model_dump()
    data["embedder"]["backend"] = "mock"
    data["embedder"]["dim"] = 768
    data["generation"]["temperature"] = 0.7
    data["generation"]["api_key_ref"] = "OPENAI_API_KEY"
    data["pipeline"]["concurrency"] = 8
    data["pipeline"]["timeout_s"] = 30.0

    result = RAGConfig.model_validate(data)
    assert result.embedder.backend == "mock"
    assert result.embedder.dim == 768
    assert result.generation.temperature == 0.7
    assert result.generation.api_key_ref == "OPENAI_API_KEY"
    assert result.pipeline.concurrency == 8
    assert result.pipeline.timeout_s == 30.0

    path = tmp_path / "edited.toml"
    path.write_text(dict_to_toml(result.model_dump(mode="json")), encoding="utf-8")
    assert RAGConfig.from_file(path) == result


# --------------------------------------------------------------------------- #
# Secret handling
# --------------------------------------------------------------------------- #


def test_api_key_ref_is_by_reference() -> None:
    """``api_key_ref`` is an env-var name, never a resolved credential."""
    cfg = RAGConfig(generation=GenerationConfig(api_key_ref="OPENAI_API_KEY"))
    dumped = cfg.model_dump(mode="json")
    assert dumped["generation"]["api_key_ref"] == "OPENAI_API_KEY"
    assert cfg.generation.api_key_ref == "OPENAI_API_KEY"


def test_mask_secret_reveals_only_tail() -> None:
    assert mask_secret("ABCDEFGH") == "•••••EFGH"
    assert mask_secret("ABCDE", keep=3) == "•••••CDE"
    assert mask_secret("ab") == "••"
    assert mask_secret("abcd") == "••••"
    assert mask_secret(None) == "<unset>"
    assert mask_secret("") == "<unset>"


# --------------------------------------------------------------------------- #
# TOML serializer structure
# --------------------------------------------------------------------------- #


def test_dict_to_toml_structure() -> None:
    toml_str = dict_to_toml(RAGConfig.mock().model_dump(mode="json"))
    assert "[pipeline]" in toml_str
    assert "[embedder]" in toml_str
    assert "[storage]" in toml_str
    assert "[generation]" in toml_str
    assert 'backend = "mock"' in toml_str


# --------------------------------------------------------------------------- #
# Path safety (validate_save_path)
# --------------------------------------------------------------------------- #


def test_validate_save_path_accepts_cwd_toml() -> None:
    """A .toml file under CWD passes validation."""
    valid = str(Path.cwd() / "safe_config.toml")
    result = validate_save_path(valid)
    assert isinstance(result, Path)
    assert result.suffix == ".toml"


def test_validate_save_path_rejects_traversal() -> None:
    """Paths containing ../ are rejected."""
    with pytest.raises(ValueError, match="traversal"):
        validate_save_path("../../etc/evil.toml")


def test_validate_save_path_rejects_non_toml() -> None:
    """Non-.toml extensions are rejected."""
    with pytest.raises(ValueError, match="toml"):
        validate_save_path("config.txt")


def test_validate_save_path_rejects_empty() -> None:
    """Empty paths are rejected."""
    with pytest.raises(ValueError, match="empty"):
        validate_save_path("")


def test_validate_save_path_rejects_outside_dirs() -> None:
    """Paths outside CWD and home are rejected."""
    with pytest.raises(ValueError, match="path must be within"):
        validate_save_path("/rag_test_outside.toml")
