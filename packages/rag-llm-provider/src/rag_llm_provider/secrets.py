"""Secret-by-reference resolution.

ADRs 0004 / 0006: secrets are never stored by value. A :class:`SecretRef`
only ever holds *where* to find a secret (an environment variable name or a
file path); the actual credential is resolved on demand by
:func:`resolve_secret` and is never persisted onto the model, emitted by
``repr``/``str``/``model_dump``/logs, or returned in API payloads.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict
from rag_core import ConfigError, RagBaseModel

SecretKind = Literal["env", "file", "none"]


class SecretRef(RagBaseModel):
    """A reference to a secret by location, never the secret itself.

    ``kind`` selects how to resolve the secret; ``ref`` is the lookup key:
    an environment variable name (``env``), a file path (``file``), or
    ``None`` (``none``). Because the model only ever holds the reference
    string, its ``repr``/``str``/``model_dump`` output is inherently safe to
    log or send to clients.
    """

    kind: SecretKind
    ref: str | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def none(cls) -> SecretRef:
        """A no-op secret reference (used by local providers, e.g. LM Studio)."""
        return cls(kind="none", ref=None)

    def describe(self) -> str | None:
        """Return the reference identifier (env var name / file path) or ``None``."""
        if self.kind == "none":
            return None
        return self.ref

    def __repr__(self) -> str:  # never emits a resolved value
        return f"SecretRef(kind={self.kind!r}, ref={self.ref!r})"


def resolve_secret(ref: SecretRef) -> str | None:
    """Resolve a :class:`SecretRef` to its concrete value.

    - ``env``: reads ``os.environ[ref]``; a missing variable raises
      :class:`~rag_core.ConfigError` with a clear message.
    - ``file``: reads and ``strip()``s the referenced text file; a missing
      file raises :class:`~rag_core.ConfigError`.
    - ``none``: returns ``None``.

    The returned string is never stored back onto the :class:`SecretRef`.
    """
    if ref.kind == "none":
        return None

    if ref.kind == "env":
        if ref.ref is None:
            raise ConfigError("env secret reference has no variable name")
        name = ref.ref
        try:
            return os.environ[name]
        except KeyError as exc:
            raise ConfigError(f"secret environment variable {name!r} is not set") from exc

    if ref.kind == "file":
        if ref.ref is None:
            raise ConfigError("file secret reference has no path")
        path = Path(ref.ref)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"secret file {path!r} does not exist") from exc
        return text.strip()

    # Defensive: SecretKind is a closed Literal, so this is unreachable.
    raise ConfigError(f"unsupported secret kind: {ref.kind!r}")


def mask_secret(value: str | None, keep: int = 4) -> str:
    """Render a secret value for display, revealing only the trailing ``keep`` chars.

    Examples
    --------
    >>> mask_secret("sk-abc1234xyz", 4)
    '...34xyz'
    >>> mask_secret(None)
    '<unset>'
    >>> mask_secret("ab")
    'ab'
    """
    if not value:
        return "<unset>"
    if keep <= 0:
        return "*" * 4
    if len(value) <= keep:
        return value
    return f"...{value[-keep:]}"


__all__ = ["SecretKind", "SecretRef", "mask_secret", "resolve_secret"]
