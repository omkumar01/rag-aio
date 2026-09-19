"""Structured, redacted logging for rag-aio built on the stdlib ``logging``.

Features:
    * JSON-line formatter (``JSONFormatter``) producing one record per line with
      a timestamp, level, logger name, message, correlation id, and a ``fields``
      dict.
    * A ``correlation_id`` ``contextvars`` cell, auto-included in every record.
    * ``redact`` / ``REDACTED`` plus a ``RedactingFilter`` that masks values whose
      attribute keys look like secrets (key|token|secret|password|authorization|api).
    * ``truncate_for_log`` and a ``log_document_content`` flag that, when off,
      truncates document-like fields so raw content is never logged by default.
    * ``setup_logging`` configuring the root logger idempotently.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import UTC, datetime

UTC = UTC

__all__ = [
    "REDACTED",
    "JSONFormatter",
    "RedactingFilter",
    "get_correlation_id",
    "get_logger",
    "redact",
    "set_correlation_id",
    "setup_logging",
    "truncate_for_log",
]

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "rag_observe_correlation_id", default=None
)

REDACTED: str = "REDACTED"

_SECRET_KEY_RE = re.compile(r"key|token|secret|password|authorization|api", re.IGNORECASE)
_CONTENT_KEY_RE = re.compile(r"content|text|body|chunk|document|page", re.IGNORECASE)

# LogRecord attribute names that cannot be passed through ``extra`` safely.
_LOGRECORD_ATTRS: frozenset[str] = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "funcName",
        "lineno",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "taskName",
        "asctime",
    }
)

# Attribute set on every handler we manage, so ``setup_logging`` can replace
# its own handlers without touching foreign ones.
_MANAGED_ATTR = "_rag_observe_managed"


def set_correlation_id(value: str | None) -> None:
    """Set the correlation id for the current task/context."""
    _correlation_id.set(value)


def get_correlation_id() -> str | None:
    """Return the current correlation id (if any)."""
    return _correlation_id.get()


def redact(value: str, keep: int = 4) -> str:
    """Mask ``value``, keeping a short ``keep``-character prefix for identification.

    The remainder is replaced by the :data:`REDACTED` sentinel. Values shorter
    than ``keep`` characters are fully redacted.
    """
    text = str(value)
    if len(text) <= keep:
        return REDACTED
    return f"{text[:keep]}{REDACTED}"


def truncate_for_log(text: str, max_chars: int = 200) -> str:
    """Truncate ``text`` to at most ``max_chars`` of content (plus a suffix)."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


class RedactingFilter(logging.Filter):
    """Filter that masks values of secret-looking keys in ``record.fields``."""

    def filter(self, record: logging.LogRecord) -> bool:
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key in list(fields.keys()):
                if isinstance(key, str) and _SECRET_KEY_RE.search(key):
                    fields[key] = redact(str(fields[key]))
        return True


class JSONFormatter(logging.Formatter):
    """Emit each record as a single JSON line."""

    def __init__(self, *, log_document_content: bool = False, max_chars: int = 200) -> None:
        super().__init__()
        self.log_document_content = log_document_content
        self.max_chars = max_chars

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            fields = dict(fields)
        elif fields is None:
            fields = {}
        else:
            fields = {"value": fields}

        if not self.log_document_content:
            for key, value in list(fields.items()):
                if isinstance(value, str) and isinstance(key, str) and _CONTENT_KEY_RE.search(key):
                    fields[key] = truncate_for_log(value, self.max_chars)

        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "fields": fields,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    *,
    json_output: bool = True,
    log_document_content: bool = False,
) -> None:
    """Configure the root logger idempotently.

    Safe to call repeatedly: previously installed managed handlers are replaced
    with a fresh one reflecting the current arguments.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    for h in list(root.handlers):
        if getattr(h, _MANAGED_ATTR, False):
            root.removeHandler(h)

    handler: logging.Handler
    if json_output:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JSONFormatter(log_document_content=log_document_content))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    handler.addFilter(RedactingFilter())
    setattr(handler, _MANAGED_ATTR, True)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger by name.

    Structured JSON output (and secret redaction) is installed by
    :func:`setup_logging`; without it the logger behaves like a plain stdlib
    logger.
    """
    return logging.getLogger(name)
