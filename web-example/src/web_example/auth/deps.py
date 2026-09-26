"""Request-scoped auth dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from web_example.db import User

__all__ = ["NotAuthenticatedError", "get_current_user", "require_admin"]


class NotAuthenticatedError(Exception):
    """Raised when a protected route is accessed without a valid session."""


async def get_current_user(request: Request) -> User:
    """Resolve the logged-in user from the signed session cookie."""
    user_id: str | None = request.session.get("user_id")
    if not user_id:
        raise NotAuthenticatedError
    maker: async_sessionmaker[AsyncSession] = request.app.state.sessionmaker
    async with maker() as session:
        user = await session.get(User, user_id)
    if user is None:
        raise NotAuthenticatedError
    return user


def require_admin(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user
