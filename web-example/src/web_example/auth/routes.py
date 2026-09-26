"""Login / signup / logout routes (session-cookie based)."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from web_example.auth import service
from web_example.auth.deps import get_current_user
from web_example.db import User

__all__ = ["router"]

router = APIRouter(tags=["auth"])


def _templates(request: Request) -> Jinja2Templates:
    return cast(Jinja2Templates, request.app.state.templates)


@router.get("/login")
async def login_page(request: Request) -> Response:
    if request.state.user is not None:
        return RedirectResponse("/chat", status_code=303)
    return _templates(request).TemplateResponse(request, "login.html", {"error": None})


@router.post("/login", response_model=None)
async def login_submit(request: Request) -> Response:
    form = await request.form()
    username = str(form.get("username", ""))
    password = str(form.get("password", ""))
    async with request.app.state.sessionmaker() as session:
        user = await service.authenticate(session, username, password)
    if user is None:
        return _templates(request).TemplateResponse(
            request,
            "login.html",
            {"error": "Invalid username or password."},
            status_code=200,
        )
    request.session["user_id"] = user.id
    return RedirectResponse("/chat", status_code=303)


@router.get("/signup")
async def signup_page(request: Request) -> Response:
    if request.state.user is not None:
        return RedirectResponse("/chat", status_code=303)
    return _templates(request).TemplateResponse(request, "signup.html", {"error": None})


@router.post("/signup")
async def signup_submit(request: Request) -> Response:
    form = await request.form()
    username = str(form.get("username", ""))
    password = str(form.get("password", ""))
    async with request.app.state.sessionmaker() as session:
        try:
            user = await service.create_user(session, username, password)
        except service.AuthError as exc:
            return _templates(request).TemplateResponse(
                request, "signup.html", {"error": str(exc)}, status_code=200
            )
    request.session["user_id"] = user.id
    return RedirectResponse("/chat", status_code=303)


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@router.get("/whoami")
async def whoami(user: Annotated[User, Depends(get_current_user)]) -> dict[str, object]:
    return {"username": user.username, "is_admin": user.is_admin}
