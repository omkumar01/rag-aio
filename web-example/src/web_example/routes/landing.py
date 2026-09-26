"""Landing page (the only route reachable without authentication)."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import Response

__all__ = ["router"]

router = APIRouter(tags=["landing"])


@router.get("/")
async def landing(request: Request) -> Response:
    return cast(Response, request.app.state.templates.TemplateResponse(request, "landing.html", {}))


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
