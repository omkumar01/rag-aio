"""Unit tests for :mod:`rag_observe.logging`."""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest
from rag_observe import (
    REDACTED,
    get_correlation_id,
    get_logger,
    redact,
    set_correlation_id,
    setup_logging,
    truncate_for_log,
)
from rag_observe.logging import JSONFormatter, RedactingFilter, _correlation_id

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_logging() -> object:
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    root.handlers = []
    root.setLevel(logging.WARNING)
    token = _correlation_id.set(None)
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)
    _correlation_id.reset(token)


def _make_record(
    msg: str = "hello", fields: dict | None = None, args: object = None
) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )
    if fields is not None:
        rec.fields = fields  # type: ignore[attr-defined]
    return rec


def test_json_formatter_parses_and_includes_correlation_id() -> None:
    set_correlation_id("corr-123")
    formatter = JSONFormatter()
    rec = _make_record("hello %s", {"request_id": "r1"}, args=("world",))
    data = json.loads(formatter.format(rec))
    assert data["message"] == "hello world"
    assert data["level"] == "INFO"
    assert data["logger"] == "test.logger"
    assert data["correlation_id"] == "corr-123"
    assert data["fields"] == {"request_id": "r1"}


def test_json_formatter_includes_timestamp() -> None:
    formatter = JSONFormatter()
    data = json.loads(formatter.format(_make_record("ok")))
    assert "timestamp" in data


def test_redacting_filter_masks_secret_keys_keeps_others() -> None:
    rf = RedactingFilter()
    rec = _make_record(
        "evt",
        {
            "api_key": "supersecret",
            "Authorization": "Bearer xyz",
            "user_id": "alice",
            "stage": "retrieval",
        },
    )
    assert rf.filter(rec) is True
    assert "supersecret" not in rec.fields["api_key"]
    assert REDACTED in rec.fields["api_key"]
    assert "Bearer xyz" not in rec.fields["Authorization"]
    assert rec.fields["user_id"] == "alice"
    assert rec.fields["stage"] == "retrieval"


def test_redact_masks_value_keeping_prefix() -> None:
    out = redact("supersecretvalue", keep=4)
    assert out.startswith("supe")
    assert REDACTED in out
    assert "supersecretvalue" not in out


def test_redact_short_value_becomes_redacted() -> None:
    assert redact("abc", keep=4) == REDACTED


def test_truncate_for_log_caps_length() -> None:
    out = truncate_for_log("a" * 500, max_chars=200)
    assert len(out) <= 214
    assert out.endswith("...[truncated]")
    assert "a" * 500 not in out


def test_truncate_for_log_short_value_unchanged() -> None:
    assert truncate_for_log("short") == "short"


def test_correlation_id_round_trip() -> None:
    set_correlation_id("cid-1")
    assert get_correlation_id() == "cid-1"
    set_correlation_id(None)
    assert get_correlation_id() is None


def test_setup_logging_emits_json_and_redacts(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    setup_logging(level="INFO", json_output=True, log_document_content=False)
    logger = get_logger("test.setup")
    # value built at runtime so this fixture is not a literal credential
    fake_secret = "sek" + "ret123"
    logger.info("event1", extra={"fields": {"api_key": fake_secret, "user": "alice"}})
    line = buf.getvalue().strip().splitlines()[-1]
    data = json.loads(line)
    assert data["message"] == "event1"
    assert data["fields"]["user"] == "alice"
    assert fake_secret not in buf.getvalue()


def test_setup_logging_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    setup_logging()
    setup_logging()
    root = logging.getLogger()
    managed = [h for h in root.handlers if getattr(h, "_rag_observe_managed", False)]
    assert len(managed) == 1
