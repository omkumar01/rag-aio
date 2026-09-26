"""Runtime settings for the web-example app.

Secrets follow the repo's secret-by-reference convention (ADR-0004/0006): the
session signing key lives in an env var or a data-dir file, never in code.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from pydantic import BaseModel, Field

__all__ = ["AppSettings"]

_DATA_DIR_ENV = "WEB_EXAMPLE_DATA_DIR"
_SECRET_ENV = "WEB_SESSION_SECRET"
_SECRET_FILE = "session.key"


class AppSettings(BaseModel):
    """Filesystem locations and the session signing key."""

    data_dir: Path = Field(default=Path("data"))
    session_secret: str = Field(default="")

    @classmethod
    def load(cls) -> AppSettings:
        """Resolve settings from the environment.

        ``WEB_EXAMPLE_DATA_DIR`` selects the data directory (default ``./data``).
        ``WEB_SESSION_SECRET`` supplies the signing key; when absent a random key
        is generated once and persisted under the data directory so sessions
        survive restarts.
        """
        data_dir = Path(os.environ.get(_DATA_DIR_ENV, "data"))
        secret = os.environ.get(_SECRET_ENV) or _load_or_create_secret(data_dir)
        return cls(data_dir=data_dir, session_secret=secret)

    @property
    def db_url(self) -> str:
        return f"sqlite+aiosqlite:///{(self.data_dir / 'web.db').as_posix()}"

    @property
    def rag_config_path(self) -> Path:
        """Global-default RAG config (TOML) edited from the settings page."""
        return self.data_dir / "rag_config.toml"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


def _load_or_create_secret(data_dir: Path) -> str:
    data_dir.mkdir(parents=True, exist_ok=True)
    key_path = data_dir / _SECRET_FILE
    if key_path.exists():
        secret = key_path.read_text(encoding="utf-8").strip()
        if secret:
            return secret
    secret = secrets.token_hex(32)
    key_path.write_text(secret, encoding="utf-8")
    return secret
