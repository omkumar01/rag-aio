"""Stable identifiers and hashes.

Identifiers are opaque lowercase hex strings. Content hashes are SHA-256 over a
canonical serialization, so the same content always produces the same hash across
processes and runs.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from pydantic import BaseModel

_HEX: str = "utf-8"


def new_id() -> str:
    """A fresh random identifier (uuid4, hex, 32 chars)."""
    return uuid.uuid4().hex


def content_hash(data: str | bytes) -> str:
    """SHA-256 hex digest of raw content. Str and bytes inputs differ by design."""
    if isinstance(data, str):
        data = data.encode(_HEX)
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(*parts: Any) -> str:
    """Deterministic identifier derived from the given parts.

    Equal parts always produce the same id, in any process.
    """
    return content_hash(_canonical(list(parts)))


def config_hash(obj: Any) -> str:
    """Order-insensitive hash of a configuration (dict, Pydantic model, JSON value)."""
    return content_hash(_canonical(obj))
