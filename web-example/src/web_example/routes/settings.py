"""Settings page routes: global defaults (admin) and per-user overrides.

The stored settings feed the RAG manager through
:func:`web_example.services.settings_store.config_provider`; after every save
the manager is asked to invalidate its per-user cache via
``app.state.rag.invalidate(user_id)`` (silently skipped when the manager is
not wired yet).
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.auth.deps import get_current_user, require_admin
from web_example.db import User
from web_example.services import settings_store
from web_example.services.settings_store import SettingsValidationError

__all__ = ["router"]

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])

CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]


def _maker(request: Request) -> async_sessionmaker[AsyncSession]:
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    return maker


def _is_htmx(request: Request) -> bool:
    return bool(request.headers.get("HX-Request"))


def _with_masks(data: dict[str, Any]) -> dict[str, Any]:
    """Add masked display copies of secret references (hints never show full refs)."""
    view = deepcopy(data)
    for section, key in (
        ("generation", "api_key_ref"),
        ("storage", "qdrant_api_key_ref"),
        ("embedder", "api_key_ref"),
        ("rerank", "api_key_ref"),
        ("ocr", "api_key_ref"),
    ):
        values = view.get(section)
        if isinstance(values, dict) and values.get(key):
            values[key + "_masked"] = settings_store.mask_ref(str(values[key]))
    return view


def _context(
    global_data: dict[str, Any],
    user_data: dict[str, Any],
    global_effective: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    return {
        "global_view": _with_masks(global_data),
        "user_view": _with_masks(user_data),
        "global_effective": global_effective,
        "error": error,
        "provider_kinds": settings_store.PROVIDER_KINDS,
        "embedder_backends": settings_store.EMBEDDER_BACKENDS,
        "cache_backends": settings_store.CACHE_BACKENDS,
        "retrieval_modes": settings_store.RETRIEVAL_MODES,
        "rerank_strategies": settings_store.RERANK_STRATEGIES,
        "ocr_engines": settings_store.OCR_ENGINES,
        "db_url_prefixes": settings_store.DB_URL_PREFIXES,
    }


async def _load_page_data(
    request: Request, user: User
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Stored global data, the user's stored overrides, and the effective global view."""
    async with _maker(request)() as session:
        global_data = await settings_store.get_global_data(session)
        user_data = await settings_store.get_user_overrides(session, user.id)
        global_effective = settings_store.config_to_web(
            await settings_store.get_global_config(session)
        )
    return global_data, user_data, global_effective


def _render_page(
    request: Request,
    user: User,
    global_data: dict[str, Any],
    user_data: dict[str, Any],
    error: str | None,
) -> Response:
    templates = request.app.state.templates
    return cast(
        Response,
        templates.TemplateResponse(
            request,
            "settings.html",
            {
                "is_admin": user.is_admin,
                **_context(global_data, user_data, {}, error),
            },
        ),
    )


def _render_partial(
    request: Request,
    template: str,
    global_data: dict[str, Any],
    user_data: dict[str, Any],
    error: str | None,
) -> Response:
    templates = request.app.state.templates
    return cast(
        Response,
        templates.TemplateResponse(request, template, _context(global_data, user_data, {}, error)),
    )


async def _invalidate(request: Request, user_id: str | None) -> None:
    """Ask the RAG manager to drop cached per-user services (no-op when absent)."""
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        return
    try:
        await rag.invalidate(user_id)
    except Exception:  # pragma: no cover - manager-specific failures are non-fatal
        logger.debug("rag.invalidate(%s) failed", user_id, exc_info=True)


@router.get("/settings")
async def settings_page(request: Request, user: CurrentUser) -> Response:
    global_data, user_data, global_effective = await _load_page_data(request, user)
    templates = request.app.state.templates
    return cast(
        Response,
        templates.TemplateResponse(
            request,
            "settings.html",
            {
                "is_admin": user.is_admin,
                **_context(global_data, user_data, _with_masks(global_effective), None),
            },
        ),
    )


@router.post("/settings/user")
async def save_user_settings(request: Request, user: CurrentUser) -> Response:
    form = await request.form()
    global_data, user_data, _ = await _load_page_data(request, user)
    try:
        data = settings_store.parse_settings_form(form)
        async with _maker(request)() as session:
            await settings_store.save_user_overrides(session, user.id, data)
    except SettingsValidationError as exc:
        if _is_htmx(request):
            return _render_partial(
                request, "partials/settings_user.html", global_data, user_data, str(exc)
            )
        return _render_page(request, user, global_data, user_data, str(exc))
    await _invalidate(request, user.id)
    if _is_htmx(request):
        _, user_data, _ = await _load_page_data(request, user)
        return _render_partial(request, "partials/settings_user.html", global_data, user_data, None)
    return RedirectResponse("/settings", status_code=303)


@router.post("/settings/global")
async def save_global_settings(request: Request, user: AdminUser) -> Response:
    form = await request.form()
    global_data, user_data, _ = await _load_page_data(request, user)
    try:
        data = settings_store.parse_settings_form(form)
        async with _maker(request)() as session:
            await settings_store.save_global_config(session, data)
    except SettingsValidationError as exc:
        if _is_htmx(request):
            return _render_partial(
                request, "partials/settings_global.html", global_data, user_data, str(exc)
            )
        return _render_page(request, user, global_data, user_data, str(exc))
    await _invalidate(request, None)
    if _is_htmx(request):
        global_data, user_data, _ = await _load_page_data(request, user)
        return _render_partial(
            request, "partials/settings_global.html", global_data, user_data, None
        )
    return RedirectResponse("/settings", status_code=303)


@router.post("/settings/user/clear")
async def clear_user_settings(request: Request, user: CurrentUser) -> Response:
    async with _maker(request)() as session:
        await settings_store.save_user_overrides(session, user.id, None)
    await _invalidate(request, user.id)
    if _is_htmx(request):
        global_data, user_data, _ = await _load_page_data(request, user)
        return _render_partial(request, "partials/settings_user.html", global_data, user_data, None)
    return RedirectResponse("/settings", status_code=303)
