"""Service-level tests for the settings store (global defaults + per-user overrides)."""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import SettingRow
from web_example.services import settings_store


async def test_get_global_config_returns_offline_defaults_when_absent(
    session: AsyncSession,
) -> None:
    config = await settings_store.get_global_config(session)
    assert config == settings_store.RAGConfig.mock()


async def test_save_and_get_global_config_roundtrip(session: AsyncSession) -> None:
    data: dict[str, Any] = {
        "generation": {
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-test",
            "api_key_ref": "OPENAI_API_KEY",
            "temperature": 0.4,
            "max_tokens": 512,
        },
        "embedder": {"backend": "mock", "dim": 128},
        "storage": {"cache_backend": "sqlite", "cache_path": "./data/cache"},
        "retrieval": {"mode": "dense", "top_k": 5, "candidate_k": 20},
    }
    config = await settings_store.save_global_config(session, data)
    assert config.generation.provider == "openai"
    assert config.generation.temperature == 0.4
    assert config.embedder.backend == "mock"
    assert config.storage.cache_backend == "sqlite"
    assert config.pipeline.retrieval.policy == "dense"
    assert config.pipeline.retrieval.overrides["top_k"] == 5

    reloaded = await settings_store.get_global_config(session)
    assert reloaded == config
    row = await session.get(SettingRow, settings_store.GLOBAL_KEY)
    assert row is not None


async def test_save_global_config_validation_error(session: AsyncSession) -> None:
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(
            session, {"generation": {"provider": "not-a-provider"}}
        )
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(session, {"storage": {"db_url": "mysql://x"}})
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(
            session, {"generation": {"api_key_ref": "sk-secret-value!"}}
        )
    row = await session.get(SettingRow, settings_store.GLOBAL_KEY)
    assert row is None


async def test_get_merged_config_overrides_sections(session: AsyncSession) -> None:
    await settings_store.save_global_config(
        session,
        {
            "generation": {"provider": "openai", "model": "global-model", "temperature": 0.2},
            "retrieval": {"mode": "hybrid", "top_k": 10},
        },
    )
    await settings_store.save_user_overrides(session, "u1", {"generation": {"temperature": 0.9}})

    merged = await settings_store.get_merged_config(session, "u1")
    assert merged.generation.provider == "openai"  # inherited from global
    assert merged.generation.model == "global-model"  # inherited from global
    assert merged.generation.temperature == 0.9  # overridden
    assert merged.pipeline.retrieval.policy == "hybrid"  # inherited
    assert merged.pipeline.retrieval.overrides["top_k"] == 10  # inherited

    other = await settings_store.get_merged_config(session, "u2")
    assert other.generation.temperature == 0.2  # untouched user sees global only


async def test_partial_user_override_validation_error(session: AsyncSession) -> None:
    await settings_store.save_global_config(
        session, {"generation": {"provider": "openai", "model": "global-model"}}
    )
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_user_overrides(session, "u1", {"retrieval": {"mode": "sparse"}})
    overrides = await settings_store.get_user_overrides(session, "u1")
    assert overrides == {}


async def test_clear_user_overrides(session: AsyncSession) -> None:
    await settings_store.save_global_config(session, {"generation": {"temperature": 0.3}})
    await settings_store.save_user_overrides(session, "u1", {"generation": {"temperature": 0.8}})
    result = await settings_store.save_user_overrides(session, "u1", None)
    assert result is None
    assert await settings_store.get_user_overrides(session, "u1") == {}
    merged = await settings_store.get_merged_config(session, "u1")
    assert merged.generation.temperature == 0.3
    row = await session.get(SettingRow, settings_store.user_key("u1"))
    assert row is None


async def test_config_provider_returns_merged_config(session: AsyncSession) -> None:
    await settings_store.save_global_config(session, {"generation": {"temperature": 0.1}})
    await settings_store.save_user_overrides(session, "u9", {"generation": {"temperature": 0.7}})
    provided = await settings_store.config_provider(session, "u9")
    assert provided.config.generation.temperature == 0.7
    provided_global = await settings_store.config_provider(session, None)
    assert provided_global.config.generation.temperature == 0.1
    assert provided.rerank.strategy == "heuristic"
    assert provided.ocr.engine == "none"


def test_parse_settings_form_maps_fields() -> None:
    form: dict[str, Any] = {
        "provider": "openai",
        "base_url": " https://api.openai.com/v1 ",
        "model": "",
        "api_key_ref": "OPENAI_API_KEY",
        "temperature": "0.7",
        "max_tokens": "1024",
        "backend": "mock",
        "dense_model": "BAAI/bge-small-en-v1.5",
        "sparse_model": "",
        "dim": "256",
        "retrieval_mode": "dense",
        "top_k": "5",
        "candidate_k": "25",
        "qdrant_path": "",
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key_ref": "QDRANT_API_KEY",
        "db_url": "sqlite+aiosqlite:///./data/rag.db",
        "cache_backend": "sqlite",
        "cache_path": "./data/cache",
    }
    data = settings_store.parse_settings_form(form)
    assert data["generation"] == {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_ref": "OPENAI_API_KEY",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    assert data["embedder"] == {
        "backend": "mock",
        "dense_model": "BAAI/bge-small-en-v1.5",
        "dim": 256,
    }
    assert data["retrieval"] == {"mode": "dense", "top_k": 5, "candidate_k": 25}
    assert data["storage"]["qdrant_url"] == "http://localhost:6333"
    assert data["storage"]["db_url"] == "sqlite+aiosqlite:///./data/rag.db"


def test_parse_settings_form_blank_fields_are_dropped() -> None:
    data = settings_store.parse_settings_form({})
    assert data == {}


def test_parse_settings_form_rejects_bad_numbers() -> None:
    with pytest.raises(settings_store.SettingsValidationError):
        settings_store.parse_settings_form({"temperature": "hot"})
    with pytest.raises(settings_store.SettingsValidationError):
        settings_store.parse_settings_form({"top_k": "many"})


def test_mask_ref_shows_last_two_chars_only() -> None:
    assert settings_store.mask_ref(None) == ""
    assert settings_store.mask_ref("") == ""
    masked = settings_store.mask_ref("OPENAI_API_KEY")
    assert masked.endswith("EY")
    assert "OPENAI" not in masked


def test_config_to_web_roundtrip() -> None:
    web = settings_store.config_to_web(settings_store.RAGConfig())
    assert web["generation"]["provider"] == "lm_studio"
    assert web["retrieval"]["mode"] == "hybrid"
    payload = settings_store.web_to_config_dict(web)
    assert settings_store.RAGConfig.model_validate(payload) == settings_store.RAGConfig()
