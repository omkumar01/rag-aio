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
    env_overrides,
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


# --------------------------------------------------------------------------- #
# RAG_AIO_ environment overrides (ADR-0006 tier: defaults -> TOML -> env)
# --------------------------------------------------------------------------- #


def test_env_overrides_nests_on_double_underscore() -> None:
    env = {
        "RAG_AIO_PIPELINE__EMBEDDING__POLICY": "mock",
        "RAG_AIO_STORAGE__DB_URL": "sqlite+aiosqlite:///./data/other.db",
    }
    assert env_overrides(env=env) == {
        "pipeline": {"embedding": {"policy": "mock"}},
        "storage": {"db_url": "sqlite+aiosqlite:///./data/other.db"},
    }


def test_env_overrides_json_coercion_and_raw_string_fallback() -> None:
    env = {
        "RAG_AIO_GENERATION__TEMPERATURE": "0.5",  # JSON number -> float
        "RAG_AIO_GENERATION__MAX_TOKENS": "512",  # JSON number -> int
        "RAG_AIO_GENERATION__BASE_URL": "http://localhost:1234/v1",  # invalid JSON -> raw str
        "RAG_AIO_EMBEDDER__BACKEND": "mock",  # invalid JSON -> raw str
    }
    assert env_overrides(env=env) == {
        "generation": {
            "temperature": 0.5,
            "max_tokens": 512,
            "base_url": "http://localhost:1234/v1",
        },
        "embedder": {"backend": "mock"},
    }


def test_env_overrides_single_segment_sets_top_level_value() -> None:
    env = {"RAG_AIO_PIPELINE": "true"}
    assert env_overrides(env=env) == {"pipeline": True}


def test_env_overrides_drops_unknown_top_level_keys() -> None:
    """Unknown top-level segments are dropped so extra="forbid" stays meaningful."""
    env = {
        "RAG_AIO_ASK__STREAM": "true",
        "RAG_AIO_NOT_A_FIELD": "1",
        "RAG_AIO_GENERATION__TEMPERATURE": "0.1",
    }
    assert env_overrides(env=env) == {"generation": {"temperature": 0.1}}


def test_env_overrides_ignores_non_prefixed_and_empty_names() -> None:
    env = {
        "OTHER_STORAGE__DB_URL": "x",
        "RAG_AIO_": "empty remainder",
        "rag_aio_generation__temperature": "0.3",  # prefix is case-sensitive
    }
    assert env_overrides(env=env) == {}


def test_env_overrides_empty_env_returns_empty_dict() -> None:
    assert env_overrides(env={}) == {}


def test_from_file_env_overrides_toml_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env wins over TOML per ADR-0006 precedence (defaults -> TOML -> env)."""
    toml_str = """\
[storage]
db_url = "sqlite+aiosqlite:///./data/from-toml.db"

[generation]
temperature = 0.7
"""
    path = tmp_path / "config.toml"
    path.write_text(toml_str, encoding="utf-8")
    monkeypatch.setenv("RAG_AIO_STORAGE__DB_URL", "sqlite+aiosqlite:///./data/from-env.db")

    cfg = RAGConfig.from_file(path)
    assert cfg.storage.db_url == "sqlite+aiosqlite:///./data/from-env.db"
    # Untouched TOML values survive the merge.
    assert cfg.generation.temperature == 0.7


def test_from_file_env_only_override_without_toml_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env var for a key absent from the TOML fills it in (env-only path)."""
    toml_str = """\
[embedder]
backend = "mock"
"""
    path = tmp_path / "minimal.toml"
    path.write_text(toml_str, encoding="utf-8")
    monkeypatch.setenv("RAG_AIO_GENERATION__BASE_URL", "http://localhost:9999/v1")

    cfg = RAGConfig.from_file(path)
    assert cfg.embedder.backend == "mock"
    assert cfg.generation.base_url == "http://localhost:9999/v1"


def test_from_file_env_json_coercion_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON-coercible env values arrive typed, not as strings."""
    path = tmp_path / "empty.toml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("RAG_AIO_GENERATION__TEMPERATURE", "0.2")

    cfg = RAGConfig.from_file(path)
    assert cfg.generation.temperature == 0.2


def test_from_file_env_unknown_top_level_key_is_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown RAG_AIO_ top-level keys are dropped, not passed to validation."""
    path = tmp_path / "empty.toml"
    path.write_text("", encoding="utf-8")
    monkeypatch.setenv("RAG_AIO_ASK__STREAM", "true")

    cfg = RAGConfig.from_file(path)  # must not raise ValidationError
    assert cfg == RAGConfig()
