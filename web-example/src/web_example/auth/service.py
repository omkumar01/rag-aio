"""User account operations (pure async service layer, no HTTP).

Lookups use explicit bound-parameter SQL (:name placeholders) so user input is
never interpolated into a statement.
"""

from __future__ import annotations

import re

import bcrypt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from web_example.db import User

__all__ = [
    "AuthError",
    "authenticate",
    "create_user",
    "get_by_username",
    "hash_password",
    "validate_credentials",
    "verify_password",
]

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")
_MIN_PASSWORD_LEN = 8


class AuthError(ValueError):
    """Raised on invalid signup input or taken usernames."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:
        return False


def validate_credentials(username: str, password: str) -> None:
    if not _USERNAME_RE.fullmatch(username):
        raise AuthError("Username must be 3-64 characters (letters, digits, '_' or '-').")
    if len(password) < _MIN_PASSWORD_LEN:
        raise AuthError("Password must be at least 8 characters.")


async def get_by_username(session: AsyncSession, username: str) -> User | None:
    row = (
        await session.execute(
            text("SELECT id FROM users WHERE username = :username"),
            {"username": username},
        )
    ).first()
    if row is None:
        return None
    return await session.get(User, row.id)


async def has_any_users(session: AsyncSession) -> bool:
    row = (
        await session.execute(
            text("SELECT COUNT(*) AS n FROM users WHERE username != :username"),
            {"username": ""},
        )
    ).scalar_one()
    return int(row) > 0


async def create_user(session: AsyncSession, username: str, password: str) -> User:
    """Create a user; the first registered account becomes admin automatically."""
    validate_credentials(username, password)
    if await get_by_username(session, username) is not None:
        raise AuthError("Username is already taken.")
    user = User(
        username=username,
        password_hash=hash_password(password),
        is_admin=not await has_any_users(session),
    )
    session.add(user)
    await session.commit()
    return user


async def authenticate(session: AsyncSession, username: str, password: str) -> User | None:
    """Return the user when the credentials match, else ``None``."""
    user = await get_by_username(session, username)
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user
