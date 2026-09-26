"""Tests for the resolved-settings contract (rerank/OCR sections, overrides)."""

from __future__ import annotations

import pytest
from rag_aio.config import GenerationConfig
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.services import settings_store
from web_example.services.local_services import ResolvedSettings
from web_example.services.rag_factory import RagManager


async def test_config_provider_returns_rerank_and_ocr(session: AsyncSession) -> None:
    await settings_store.save_global_config(
        session,
        {
            "rerank": {
                "strategy": "remote",
                "model": "bge-reranker-base",
                "base_url": "http://localhost:1234/v1",
                "api_key_ref": "RERANK_API_KEY",
                "top_k": 3,
            },
            "ocr": {
                "engine": "vlm",
                "base_url": "http://localhost:1234/v1",
                "model": "my-ocr",
                "api_key_ref": "OCR_API_KEY",
            },
            "storage": {"qdrant_url": "http://localhost:6333"},
        },
    )
    resolved = await settings_store.config_provider(session, None)
    assert isinstance(resolved, ResolvedSettings)
    assert resolved.rerank.strategy == "remote"
    assert resolved.rerank.model == "bge-reranker-base"
    assert resolved.rerank.api_key_ref == "RERANK_API_KEY"
    assert resolved.ocr.engine == "vlm"
    assert resolved.ocr.model == "my-ocr"
    assert resolved.storage_extras.qdrant_url == "http://localhost:6333"


async def test_user_can_override_rerank_strategy(session: AsyncSession) -> None:
    await settings_store.save_global_config(session, {"rerank": {"strategy": "heuristic"}})
    await settings_store.save_user_overrides(
        session, "u-1", {"rerank": {"strategy": "remote", "base_url": "http://x/v1"}}
    )
    resolved = await settings_store.config_provider(session, "u-1")
    assert resolved.rerank.strategy == "remote"
    assert resolved.rerank.base_url == "http://x/v1"

    other = await settings_store.config_provider(session, "u-2")
    assert other.rerank.strategy == "heuristic"


async def test_rerank_validation_rejects_secret_value(session: AsyncSession) -> None:
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(
            session, {"rerank": {"strategy": "remote", "api_key_ref": "sk-real-secret"}}
        )


def test_parse_settings_form_reads_rerank_and_ocr() -> None:
    parsed = settings_store.parse_settings_form(
        {
            "rerank_strategy": "cross_encoder",
            "rerank_model": "my-model",
            "ocr_engine": "vlm",
            "ocr_model": "ocr-1",
        }
    )
    assert parsed["rerank"] == {"strategy": "cross_encoder", "model": "my-model"}
    assert parsed["ocr"] == {"engine": "vlm", "model": "ocr-1"}


def test_pipeline_config_carries_generation_overrides() -> None:
    config = settings_store.RAGConfig(
        generation=GenerationConfig(model="my-model", temperature=0.7, max_tokens=512)
    )
    pipeline = RagManager._pipeline_config(config)
    overrides = pipeline.generation.overrides
    assert overrides["model"] == "my-model"
    assert overrides["temperature"] == 0.7
    assert overrides["max_tokens"] == 512


async def test_remote_embedder_backend_roundtrip(session: AsyncSession) -> None:
    await settings_store.save_global_config(
        session,
        {
            "embedder": {
                "backend": "remote",
                "dense_model": "text-embedding-nomic-embed-text-v1.5",
                "dim": 768,
                "base_url": "http://localhost:1234/v1",
                "api_key_ref": "EMBEDDING_API_KEY",
            }
        },
    )
    resolved = await settings_store.config_provider(session, None)
    assert resolved.config.embedder.backend == "remote"
    assert resolved.config.embedder.dense_model == "text-embedding-nomic-embed-text-v1.5"
    assert resolved.config.embedder.dim == 768
    assert resolved.embedder_extras.base_url == "http://localhost:1234/v1"
    assert resolved.embedder_extras.api_key_ref == "EMBEDDING_API_KEY"


async def test_remote_embedder_validation(session: AsyncSession) -> None:
    base = {
        "backend": "remote",
        "dense_model": "nomic",
        "dim": 768,
        "base_url": "http://localhost:1234/v1",
    }
    # Valid save succeeds.
    await settings_store.save_global_config(session, {"embedder": dict(base)})

    missing_model = dict(base, dense_model="")
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(session, {"embedder": missing_model})

    missing_dim = dict(base, dim="")
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(session, {"embedder": missing_dim})

    missing_url = dict(base, base_url="")
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(session, {"embedder": missing_url})

    bad_url = dict(base, base_url="not-a-url")
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(session, {"embedder": bad_url})

    secret_ref = dict(base, api_key_ref="sk-live-secret")
    with pytest.raises(settings_store.SettingsValidationError):
        await settings_store.save_global_config(session, {"embedder": secret_ref})


def test_parse_settings_form_reads_embedder_extras() -> None:
    parsed = settings_store.parse_settings_form(
        {
            "backend": "remote",
            "embedder_base_url": "http://localhost:1234/v1",
            "embedder_api_key_ref": "EMBEDDING_API_KEY",
        }
    )
    assert parsed["embedder"] == {
        "backend": "remote",
        "base_url": "http://localhost:1234/v1",
        "api_key_ref": "EMBEDDING_API_KEY",
    }
