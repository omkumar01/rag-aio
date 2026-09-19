"""Tests for secret-by-reference resolution and masking."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from rag_llm_provider import ConfigError, SecretRef, mask_secret, resolve_secret

# --- env ---


def test_resolve_env_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TEST_API_KEY", "sk-actual-secret-value")
    ref = SecretRef(kind="env", ref="RAG_TEST_API_KEY")
    assert resolve_secret(ref) == "sk-actual-secret-value"


def test_resolve_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_TEST_MISSING_ENV", raising=False)
    ref = SecretRef(kind="env", ref="RAG_TEST_MISSING_ENV")
    with pytest.raises(ConfigError, match="RAG_TEST_MISSING_ENV"):
        resolve_secret(ref)


def test_resolve_env_with_no_ref_raises() -> None:
    ref = SecretRef(kind="env", ref=None)
    with pytest.raises(ConfigError, match="variable name"):
        resolve_secret(ref)


# --- file ---


def test_resolve_file_secret(tmp_path: Path) -> None:
    secret_file = tmp_path / "key.txt"
    secret_file.write_text("  file-secret-value\n", encoding="utf-8")
    ref = SecretRef(kind="file", ref=str(secret_file))
    assert resolve_secret(ref) == "file-secret-value"


def test_resolve_missing_file_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nope.txt"
    ref = SecretRef(kind="file", ref=str(missing))
    with pytest.raises(ConfigError, match="does not exist"):
        resolve_secret(ref)


def test_resolve_file_with_no_ref_raises() -> None:
    ref = SecretRef(kind="file", ref=None)
    with pytest.raises(ConfigError, match="no path"):
        resolve_secret(ref)


# --- none ---


def test_resolve_none_secret() -> None:
    ref = SecretRef(kind="none", ref=None)
    assert resolve_secret(ref) is None


# --- masking ---


def test_mask_secret_keeps_last_four() -> None:
    assert mask_secret("abcdef1234567890", keep=4) == "...7890"


def test_mask_secret_unset() -> None:
    assert mask_secret(None) == "<unset>"
    assert mask_secret("") == "<unset>"


def test_mask_secret_short_value_unchanged() -> None:
    # shorter than or equal to keep: shown in full (not enough to mask)
    assert mask_secret("ab", keep=4) == "ab"
    assert mask_secret("abcd", keep=4) == "abcd"


# --- repr / safety ---


def test_secretref_never_holds_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # A SecretRef only ever holds the *reference* (env var name); the resolved
    # value is never stored on the model and never reachable via repr/str/dump.
    monkeypatch.setenv("MY_API_KEY", "sk-actual-secret-value")
    ref = SecretRef(kind="env", ref="MY_API_KEY")
    assert ref.describe() == "MY_API_KEY"
    # repr must not surface a resolved value (it only ever shows the ref name)
    assert "MY_API_KEY" in repr(ref)
    assert "sk-actual-secret-value" not in repr(ref)
    assert ref.kind == "env"
    assert ref.ref == "MY_API_KEY"
    # resolving does not mutate the ref back onto the model
    resolved = resolve_secret(ref)
    assert resolved == "sk-actual-secret-value"
    assert ref.ref == "MY_API_KEY"


def test_secretref_forbids_extra() -> None:
    with pytest.raises(ValidationError):
        SecretRef(kind="env", ref="X", value="should-not-be-allowed")


def test_secretref_invalid_kind() -> None:
    with pytest.raises(ValidationError):
        SecretRef(kind="bogus", ref="x")  # type: ignore[arg-type]
