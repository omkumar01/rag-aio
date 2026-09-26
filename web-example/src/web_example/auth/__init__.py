"""Authentication: bcrypt password hashing, session-backed users."""

from web_example.auth.deps import NotAuthenticatedError, get_current_user, require_admin
from web_example.auth.service import authenticate, create_user, hash_password, verify_password

__all__ = [
    "NotAuthenticatedError",
    "authenticate",
    "create_user",
    "get_current_user",
    "hash_password",
    "require_admin",
    "verify_password",
]
