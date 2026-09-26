"""FastAPI application factory for the rag-aio web example."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import RequestResponseEndpoint
from starlette.middleware.sessions import SessionMiddleware

from web_example import __version__
from web_example.auth.deps import NotAuthenticatedError
from web_example.db import User, init_db
from web_example.routes import ROUTERS
from web_example.services.rag_factory import RagManager
from web_example.services.settings_store import config_provider as rag_config_provider
from web_example.settings import AppSettings

__all__ = ["create_app"]

_PKG_DIR = Path(__file__).resolve().parent


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Build the web-example app over the rag-aio packages."""
    settings = settings or AppSettings.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings.ensure_dirs()
        if getattr(app.state, "rag", None) is None:
            app.state.rag = RagManager(settings)
        await init_db(app)
        yield
        await app.state.engine.dispose()

    app = FastAPI(
        title="rag-aio web example",
        version=__version__,
        description="Web UI over the rag-aio packages (chat, documents, settings, bulk ingest).",
        lifespan=lifespan,
    )
    app.state.settings = settings
    # RAG engine: resolves each user's merged config (global defaults + per-user
    # overrides stored by the settings page); mock profile until configured
    # otherwise. Heavy backends are only imported when a non-mock config resolves.
    app.state.rag = RagManager(settings, config_provider=rag_config_provider)
    app.mount(
        "/static",
        StaticFiles(directory=str(_PKG_DIR / "static")),
        name="static",
    )
    app.state.templates = Jinja2Templates(directory=str(_PKG_DIR / "templates"))

    @app.middleware("http")
    async def attach_user(request: Request, call_next: RequestResponseEndpoint) -> Response:
        request.state.user = None
        user_id: str | None = request.session.get("user_id")
        if user_id:
            async with request.app.state.sessionmaker() as session:
                request.state.user = await session.get(User, user_id)
        return await call_next(request)

    for router in ROUTERS:
        app.include_router(router)

    @app.exception_handler(NotAuthenticatedError)
    async def not_authenticated(request: Request, exc: NotAuthenticatedError) -> Response:
        if request.headers.get("HX-Request"):
            return HTMLResponse(status_code=200, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)

    # Added last so it sits outermost and request.session is available to the
    # attach_user middleware and every route.
    app.add_middleware(SessionMiddleware, secret_key=settings.session_secret, same_site="lax")

    return app


# Module-level app for `uvicorn web_example.main:app --app-dir web-example/src`.
app = create_app()
