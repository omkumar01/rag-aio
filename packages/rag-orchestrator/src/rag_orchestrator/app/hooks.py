"""Auth/tenant hooks for the FastAPI service surface.

These abstractions let a host process inject per-request authentication and
tenant -> namespace mapping into :func:`rag_orchestrator.create_app` without
coupling the app to any identity provider. They deliberately mirror the
protocol style of :mod:`rag_core.protocols` (ADR-0002).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import Field
from rag_core.base import RagBaseModel

__all__ = ["AuthContext", "NamespaceResolver", "RequestAuthenticator"]


class AuthContext(RagBaseModel):
    """Identity of an authenticated request.

    Attributes:
        subject: Stable identifier for the caller (user/service principal).
        tenant: Optional tenant identifier for multi-tenant deployments.
        namespace: Optional namespace hint derived from the identity.
        attributes: Free-form claims/attributes from the authenticator.
    """

    subject: str
    tenant: str | None = None
    namespace: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class RequestAuthenticator(Protocol):
    """Authenticates an incoming request; return ``None`` to reject (401)."""

    async def authenticate(self, request: Any) -> AuthContext | None: ...


@runtime_checkable
class NamespaceResolver(Protocol):
    """Maps an :class:`AuthContext` to the namespace its data lives in.

    When wired together with a :class:`RequestAuthenticator` the resolved
    namespace wins over any client-supplied ``namespace`` in the request body,
    so tenant isolation cannot be overridden by callers.
    """

    def namespace_for(self, auth: AuthContext) -> str | None: ...
