"""Settings persistence: global RAG defaults and per-user overrides.

Settings live as JSON blobs in :class:`~web_example.db.SettingRow`:

* key ``"global"`` — the admin-authored global default configuration,
* key ``"user:<user_id>"`` — one user's *partial* overrides.

The stored shape ("web view") has top-level sections — ``generation``,
``embedder``, ``storage``, ``retrieval``, ``rerank`` and ``ocr`` — holding only
the fields the settings form exposes. :func:`web_to_config_dict` maps a web
view onto a :class:`~rag_aio.config.RAGConfig` payload (``retrieval`` becomes
``pipeline.retrieval``; ``rerank``/``ocr`` are web-only options consumed by
:mod:`~web_example.services.local_services`); qdrant *server* fields
(``qdrant_url`` / ``qdrant_api_key_ref``) are kept for the UI round-trip but
are not part of the current ``StorageConfig`` schema, so they are preserved
verbatim without being validated into the model.

Secrets follow the repo's secret-by-reference convention: ``api_key_ref`` /
``qdrant_api_key_ref`` hold environment-variable *names*, never values.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydantic import ValidationError
from rag_aio.config import RAGConfig
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import SettingRow
from web_example.services.local_services import (
    EmbedderExtras,
    OcrSettings,
    RerankSettings,
    ResolvedSettings,
    StorageExtras,
)

__all__ = [
    "CACHE_BACKENDS",
    "DB_URL_PREFIXES",
    "EMBEDDER_BACKENDS",
    "GLOBAL_KEY",
    "PROVIDER_KINDS",
    "RETRIEVAL_MODES",
    "SettingsValidationError",
    "config_provider",
    "config_to_web",
    "get_global_config",
    "get_global_data",
    "get_merged_config",
    "get_user_overrides",
    "mask_ref",
    "merge_data",
    "parse_settings_form",
    "save_global_config",
    "save_user_overrides",
    "user_key",
    "validate_web_data",
    "web_to_config_dict",
]

#: SettingRow key holding the global default configuration.
GLOBAL_KEY = "global"

#: Allowed generation provider kinds (see rag_llm_provider.models_info.ProviderKind).
PROVIDER_KINDS: tuple[str, ...] = (
    "openai",
    "anthropic",
    "gemini",
    "openai_compatible",
    "ollama",
    "vllm",
    "llamacpp",
)

EMBEDDER_BACKENDS: tuple[str, ...] = ("mock", "fastembed", "remote")
CACHE_BACKENDS: tuple[str, ...] = ("memory", "sqlite")

#: Retrieval strategy modes exposed on the settings form.
RETRIEVAL_MODES: tuple[str, ...] = ("hybrid", "dense")

#: Rerank strategies and OCR engines exposed on the settings form.
RERANK_STRATEGIES: tuple[str, ...] = ("heuristic", "cross_encoder", "remote")
OCR_ENGINES: tuple[str, ...] = ("none", "vlm")

#: Allowed SQL document-store URL schemes.
DB_URL_PREFIXES: tuple[str, ...] = ("sqlite+aiosqlite", "postgresql+asyncpsycopg")

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_SECTION_KEYS: dict[str, frozenset[str]] = {
    "generation": frozenset(
        {"provider", "base_url", "model", "api_key_ref", "temperature", "max_tokens"}
    ),
    "embedder": frozenset(
        {"backend", "dense_model", "sparse_model", "dim", "base_url", "api_key_ref"}
    ),
    "storage": frozenset({"qdrant_path", "db_url", "cache_backend", "cache_path"}),
    "rerank": frozenset({"strategy", "model", "base_url", "api_key_ref", "top_k"}),
    "ocr": frozenset({"engine", "base_url", "model", "api_key_ref"}),
}

#: Storage fields kept for the UI (qdrant server mode) but absent from StorageConfig.
_WEB_ONLY_STORAGE_KEYS: frozenset[str] = frozenset({"qdrant_url", "qdrant_api_key_ref"})

#: Embedder fields kept for the UI (remote backend) but absent from EmbedderConfig.
_WEB_ONLY_EMBEDDER_KEYS: frozenset[str] = frozenset({"base_url", "api_key_ref"})

_RETRIEVAL_KEYS: frozenset[str] = frozenset({"mode", "top_k", "candidate_k"})

_API_KEY_FIELDS: dict[str, str] = {
    "generation": "api_key_ref",
    "storage": "qdrant_api_key_ref",
    "embedder": "api_key_ref",
    "rerank": "api_key_ref",
    "ocr": "api_key_ref",
}


class SettingsValidationError(ValueError):
    """Raised when settings data cannot be validated into a :class:`RAGConfig`."""


def user_key(user_id: str) -> str:
    """SettingRow key for a user's overrides."""
    return "user:" + user_id


# -- raw storage helpers -----------------------------------------------------


async def _load_data(session: AsyncSession, key: str) -> dict[str, Any]:
    row = await session.get(SettingRow, key)
    if row is None or not isinstance(row.data, dict):
        return {}
    return deepcopy(row.data)


async def get_global_data(session: AsyncSession) -> dict[str, Any]:
    """Stored global web-view dict (``{}`` when nothing was saved yet)."""
    return await _load_data(session, GLOBAL_KEY)


async def get_user_overrides(session: AsyncSession, user_id: str) -> dict[str, Any]:
    """Stored per-user override web-view dict (``{}`` when the user inherits everything)."""
    return await _load_data(session, user_key(user_id))


async def _write_data(session: AsyncSession, key: str, data: dict[str, Any]) -> None:
    row = await session.get(SettingRow, key)
    if row is None:
        session.add(SettingRow(key=key, data=data))
    else:
        row.data = data
    await session.commit()


async def _delete_data(session: AsyncSession, key: str) -> None:
    row = await session.get(SettingRow, key)
    if row is not None:
        await session.delete(row)
        await session.commit()


# -- web view <-> RAGConfig payload ------------------------------------------


def _clean_section(data: Mapping[str, Any], section: str) -> dict[str, Any]:
    values = data.get(section)
    if not isinstance(values, Mapping):
        return {}
    web_only: frozenset[str] = frozenset()
    if section == "storage":
        web_only = _WEB_ONLY_STORAGE_KEYS
    elif section == "embedder":
        web_only = _WEB_ONLY_EMBEDDER_KEYS
    allowed = _SECTION_KEYS[section] | web_only
    return {k: v for k, v in values.items() if k in allowed and v is not None and v != ""}


def merge_data(global_data: Mapping[str, Any], override_data: Mapping[str, Any]) -> dict[str, Any]:
    """Overlay *override_data* onto *global_data*, key-by-key within each section.

    Sections absent from the overrides are inherited whole; within an overridden
    section only the keys the user actually set replace the global ones, so
    per-user overrides stay partial.
    """
    merged = deepcopy(dict(global_data))
    for section, values in override_data.items():
        if isinstance(values, Mapping):
            base = merged.get(section)
            base = dict(base) if isinstance(base, Mapping) else {}
            base.update(values)
            merged[section] = base
        else:
            merged[section] = values
    return merged


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (override wins at leaf level)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _effective_backend(data: Mapping[str, Any]) -> str:
    """The configured embedder backend; ``mock`` unless a real one was chosen.

    Offline-by-default: with nothing stored (or ``backend=mock``) the merged
    config uses the fully-offline mock profile; selecting ``fastembed`` or
    ``remote`` in the embedder section switches to the real pipeline.
    """
    embedder = data.get("embedder")
    if isinstance(embedder, Mapping) and embedder.get("backend") in ("fastembed", "remote"):
        return str(embedder["backend"])
    return "mock"


def web_to_config_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Map a web-view dict onto a :class:`RAGConfig` model payload.

    ``retrieval`` becomes ``pipeline.retrieval`` (``mode`` -> ``policy`` plus
    ``overrides``); web-only storage keys are dropped from the payload. The
    payload is deep-merged over the built-in *offline* profile's dump
    (:meth:`RAGConfig.mock`) so sections the form does not touch keep sensible
    defaults (``PipelineConfig.local_default`` stages would otherwise reset to
    bare ``StageConfig`` defaults once a single pipeline key is present), and
    the app stays fully offline until a non-mock embedder is configured.
    """
    payload: dict[str, Any] = {}
    for section in ("generation", "embedder", "storage"):
        cleaned = _clean_section(data, section)
        if cleaned:
            web_only = (
                _WEB_ONLY_STORAGE_KEYS
                if section == "storage"
                else _WEB_ONLY_EMBEDDER_KEYS
                if section == "embedder"
                else frozenset()
            )
            payload[section] = {k: v for k, v in cleaned.items() if k not in web_only}
    retrieval = data.get("retrieval")
    if isinstance(retrieval, Mapping):
        overrides: dict[str, Any] = {
            k: v for k, v in retrieval.items() if k in _RETRIEVAL_KEYS - {"mode"} and v is not None
        }
        mode = retrieval.get("mode")
        stage: dict[str, Any] = {}
        if mode:
            stage["policy"] = mode
            overrides["mode"] = mode
        if stage or overrides:
            stage["overrides"] = overrides
            payload["pipeline"] = {"retrieval": stage}
    base = RAGConfig() if _effective_backend(data) in ("fastembed", "remote") else RAGConfig.mock()
    return _deep_merge(base.model_dump(mode="python"), payload)


def config_to_web(config: RAGConfig) -> dict[str, Any]:
    """Inverse of :func:`web_to_config_dict` for a fully-resolved config (UI display)."""
    retrieval = config.pipeline.retrieval
    mode = retrieval.overrides.get("mode") or retrieval.policy
    return {
        "generation": {
            "provider": config.generation.provider,
            "base_url": config.generation.base_url,
            "model": config.generation.model,
            "api_key_ref": config.generation.api_key_ref,
            "temperature": config.generation.temperature,
            "max_tokens": config.generation.max_tokens,
        },
        "embedder": {
            "backend": config.embedder.backend,
            "dense_model": config.embedder.dense_model,
            "sparse_model": config.embedder.sparse_model,
            "dim": config.embedder.dim,
            "base_url": "",
            "api_key_ref": "",
        },
        "retrieval": {
            "mode": mode,
            "top_k": retrieval.overrides.get("top_k"),
            "candidate_k": retrieval.overrides.get("candidate_k"),
        },
        "storage": {
            "qdrant_path": config.storage.qdrant_path,
            "db_url": config.storage.db_url,
            "cache_backend": config.storage.cache_backend,
            "cache_path": config.storage.cache_path,
            "qdrant_url": "",
            "qdrant_api_key_ref": "",
        },
        "rerank": {
            "strategy": "heuristic",
            "model": "",
            "base_url": "",
            "api_key_ref": "",
            "top_k": "",
        },
        "ocr": {
            "engine": "none",
            "base_url": "http://localhost:1234/v1",
            "model": "glm-ocr",
            "api_key_ref": "",
        },
    }


# -- validation --------------------------------------------------------------


def _require_choice(section_values: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> None:
    value = section_values.get(key)
    if value is not None and value not in allowed:
        raise SettingsValidationError(f"{key} must be one of: {', '.join(allowed)}.")


def _require_env_name(section_values: Mapping[str, Any], key: str) -> None:
    ref = section_values.get(key)
    if ref and not _ENV_NAME_RE.match(str(ref)):
        raise SettingsValidationError(
            f"{key} must be an environment variable NAME (e.g. OPENAI_API_KEY), "
            "never a secret value."
        )


def _require_int(section_values: Mapping[str, Any], key: str, minimum: int = 1) -> None:
    value = section_values.get(key)
    if value is None:
        return
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SettingsValidationError(f"{key} must be an integer.") from exc
    if number < minimum:
        raise SettingsValidationError(f"{key} must be >= {minimum}.")


def _check_web_data(data: Mapping[str, Any]) -> None:
    for section, values in data.items():
        if section not in _SECTION_KEYS and section != "retrieval":
            raise SettingsValidationError(f"Unknown settings section: {section}.")
        if not isinstance(values, Mapping):
            raise SettingsValidationError(f"{section} must be an object.")

    generation = data.get("generation") or {}
    _require_choice(generation, "provider", PROVIDER_KINDS)
    _require_env_name(generation, "api_key_ref")

    embedder = data.get("embedder") or {}
    _require_choice(embedder, "backend", EMBEDDER_BACKENDS)
    _require_int(embedder, "dim")
    _require_env_name(embedder, "api_key_ref")
    if embedder.get("backend") == "remote":
        # The vector store is sized up-front, so a remote embedder must declare
        # its dimension, and /embeddings needs a model id and endpoint.
        if not embedder.get("dense_model"):
            raise SettingsValidationError(
                "remote embedding requires a model (dense model id, e.g. "
                "text-embedding-nomic-embed-text-v1.5)."
            )
        if not embedder.get("dim"):
            raise SettingsValidationError(
                "remote embedding requires an explicit dimension (the vector "
                "store is sized before the first response)."
            )
        base_url = embedder.get("base_url")
        if base_url and not str(base_url).startswith(("http://", "https://")):
            raise SettingsValidationError("embedder base_url must be an http(s) server URL.")
        if not base_url:
            raise SettingsValidationError("remote embedding requires a base_url endpoint.")

    storage = data.get("storage") or {}
    _require_choice(storage, "cache_backend", CACHE_BACKENDS)
    _require_env_name(storage, "qdrant_api_key_ref")
    db_url = storage.get("db_url")
    if db_url and not str(db_url).startswith(DB_URL_PREFIXES):
        raise SettingsValidationError(
            f"db_url must start with one of: {', '.join(DB_URL_PREFIXES)}."
        )
    qdrant_url = storage.get("qdrant_url")
    if qdrant_url and not str(qdrant_url).startswith(("http://", "https://")):
        raise SettingsValidationError("qdrant_url must be an http(s) server URL.")
    if storage.get("qdrant_path") and qdrant_url:
        raise SettingsValidationError(
            "Set either a local qdrant path or a qdrant server url, not both."
        )

    retrieval = data.get("retrieval") or {}
    _require_choice(retrieval, "mode", RETRIEVAL_MODES)
    _require_int(retrieval, "top_k")
    _require_int(retrieval, "candidate_k")

    rerank = data.get("rerank") or {}
    _require_choice(rerank, "strategy", RERANK_STRATEGIES)
    _require_env_name(rerank, "api_key_ref")
    _require_int(rerank, "top_k")
    if (
        rerank.get("strategy") == "remote"
        and rerank.get("base_url")
        and not str(rerank["base_url"]).startswith(("http://", "https://"))
    ):
        raise SettingsValidationError("rerank base_url must be an http(s) server URL.")

    ocr = data.get("ocr") or {}
    _require_choice(ocr, "engine", OCR_ENGINES)
    _require_env_name(ocr, "api_key_ref")


def validate_web_data(data: Mapping[str, Any]) -> RAGConfig:
    """Validate a web-view dict (field checks, then ``RAGConfig.model_validate``)."""
    _check_web_data(data)
    try:
        return RAGConfig.model_validate(web_to_config_dict(data))
    except ValidationError as exc:
        details = "; ".join(
            "{}: {}".format(".".join(str(part) for part in error["loc"]), error["msg"])
            for error in exc.errors()[:5]
        )
        raise SettingsValidationError(f"Invalid settings: {details}") from exc


# -- public read/write API ---------------------------------------------------


async def get_global_config(session: AsyncSession) -> RAGConfig:
    """Global default config; built-in :class:`RAGConfig` defaults when nothing is stored."""
    return validate_web_data(await get_global_data(session))


async def get_merged_config(session: AsyncSession, user_id: str | None) -> RAGConfig:
    """Effective config for *user_id*: global defaults overlaid with their overrides.

    ``user_id=None`` (or a user with no overrides) returns the global config.
    """
    global_data = await get_global_data(session)
    if user_id is None:
        return validate_web_data(global_data)
    overrides = await get_user_overrides(session, user_id)
    return validate_web_data(merge_data(global_data, overrides))


async def save_global_config(session: AsyncSession, data: Mapping[str, Any]) -> RAGConfig:
    """Validate and persist the global defaults; returns the saved config."""
    cleaned: dict[str, Any] = {s: _clean_section(data, s) for s in _SECTION_KEYS}
    retrieval = data.get("retrieval")
    if isinstance(retrieval, Mapping):
        cleaned["retrieval"] = {k: v for k, v in retrieval.items() if k in _RETRIEVAL_KEYS}
    config = validate_web_data(cleaned)
    await _write_data(session, GLOBAL_KEY, cleaned)
    return config


async def save_user_overrides(
    session: AsyncSession, user_id: str, data: Mapping[str, Any] | None
) -> RAGConfig | None:
    """Validate (against global + overrides) and persist per-user overrides.

    ``data=None`` clears the user's overrides and returns ``None``; an empty
    dict also clears (everything inherits the global defaults).
    """
    if data is None:
        await _delete_data(session, user_key(user_id))
        return None
    cleaned: dict[str, Any] = {s: _clean_section(data, s) for s in _SECTION_KEYS}
    retrieval = data.get("retrieval")
    if isinstance(retrieval, Mapping):
        cleaned["retrieval"] = {k: v for k, v in retrieval.items() if k in _RETRIEVAL_KEYS}
    cleaned = {section: values for section, values in cleaned.items() if values}
    global_data = await get_global_data(session)
    validate_web_data(merge_data(global_data, cleaned))
    if cleaned:
        await _write_data(session, user_key(user_id), cleaned)
    else:
        await _delete_data(session, user_key(user_id))
    return None if not cleaned else await get_merged_config(session, user_id)


async def config_provider(session: AsyncSession, user_id: str | None) -> ResolvedSettings:
    """Effective-settings provider for the RagManager ``config_provider`` hook.

    Contract: ``async (session, user_id) -> ResolvedSettings`` — called with a
    request-scoped session and the effective user id (``None`` for global-only
    contexts); returns the merged (global + per-user) configuration plus the
    web-only rerank/OCR/storage-server options.
    """
    global_data = await get_global_data(session)
    data = (
        global_data
        if user_id is None
        else merge_data(global_data, await get_user_overrides(session, user_id))
    )
    storage = data.get("storage") or {}
    embedder = data.get("embedder") or {}
    rerank = data.get("rerank") or {}
    ocr = data.get("ocr") or {}
    return ResolvedSettings(
        config=validate_web_data(data),
        storage_extras=StorageExtras(
            qdrant_url=str(storage.get("qdrant_url") or "") or None,
            qdrant_api_key_ref=str(storage.get("qdrant_api_key_ref") or "") or None,
        ),
        embedder_extras=EmbedderExtras(
            base_url=str(embedder.get("base_url") or "") or None,
            api_key_ref=str(embedder.get("api_key_ref") or "") or None,
        ),
        rerank=RerankSettings.model_validate(rerank),
        ocr=OcrSettings.model_validate(ocr),
    )


# -- form parsing & display helpers ------------------------------------------


def parse_settings_form(form: Any) -> dict[str, Any]:
    """Map a settings form (starlette FormData or plain mapping) onto a web-view dict.

    Blank/absent fields are omitted entirely (blank = inherit global). Numeric
    fields are converted; malformed numbers raise :class:`SettingsValidationError`.
    """

    def raw(name: str) -> str:
        value = form.get(name)
        return str(value).strip() if value is not None else ""

    def opt_str(name: str) -> str | None:
        return raw(name) or None

    def opt_float(name: str) -> float | None:
        value = raw(name)
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise SettingsValidationError(f"{name} must be a number.") from exc

    def opt_int(name: str) -> int | None:
        value = raw(name)
        if not value:
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise SettingsValidationError(f"{name} must be an integer.") from exc

    data: dict[str, Any] = {
        "generation": {
            "provider": opt_str("provider"),
            "base_url": opt_str("base_url"),
            "model": opt_str("model"),
            "api_key_ref": opt_str("api_key_ref"),
            "temperature": opt_float("temperature"),
            "max_tokens": opt_int("max_tokens"),
        },
        "embedder": {
            "backend": opt_str("backend"),
            "dense_model": opt_str("dense_model"),
            "sparse_model": opt_str("sparse_model"),
            "dim": opt_int("dim"),
            "base_url": opt_str("embedder_base_url"),
            "api_key_ref": opt_str("embedder_api_key_ref"),
        },
        "storage": {
            "qdrant_path": opt_str("qdrant_path"),
            "qdrant_url": opt_str("qdrant_url"),
            "qdrant_api_key_ref": opt_str("qdrant_api_key_ref"),
            "db_url": opt_str("db_url"),
            "cache_backend": opt_str("cache_backend"),
            "cache_path": opt_str("cache_path"),
        },
        "retrieval": {
            "mode": opt_str("retrieval_mode"),
            "top_k": opt_int("top_k"),
            "candidate_k": opt_int("candidate_k"),
        },
        "rerank": {
            "strategy": opt_str("rerank_strategy"),
            "model": opt_str("rerank_model"),
            "base_url": opt_str("rerank_base_url"),
            "api_key_ref": opt_str("rerank_api_key_ref"),
            "top_k": opt_int("rerank_top_k"),
        },
        "ocr": {
            "engine": opt_str("ocr_engine"),
            "base_url": opt_str("ocr_base_url"),
            "model": opt_str("ocr_model"),
            "api_key_ref": opt_str("ocr_api_key_ref"),
        },
    }
    parsed: dict[str, Any] = {}
    for section, values in data.items():
        kept = {k: v for k, v in values.items() if v is not None}
        if kept:
            parsed[section] = kept
    return parsed


def mask_ref(ref: str | None) -> str:
    """Mask a secret reference for display: last 2 chars only, when set."""
    if not ref:
        return ""
    return "\u2022\u2022" + ref[-2:]
