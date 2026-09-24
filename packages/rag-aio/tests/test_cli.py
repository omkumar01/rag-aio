"""Tests for the rag-aio CLI (config, ingest, ask commands)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_aio.cli import app
from typer.testing import CliRunner

runner = CliRunner()


# --------------------------------------------------------------------------- #
# config command
# --------------------------------------------------------------------------- #


def test_config_command_outputs_json() -> None:
    result = runner.invoke(app, ["config"])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["embedder"]["backend"] == "mock"
    assert data["pipeline"]["name"] == "local_fast"
    assert "storage" in data
    assert "generation" in data


def test_config_command_with_file(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[generation]\nmodel = "test-model"\ntemperature = 0.5\n', encoding="utf-8")
    result = runner.invoke(app, ["config", "--config", str(path)])
    assert result.exit_code == 0, result.stdout
    data = json.loads(result.stdout)
    assert data["generation"]["model"] == "test-model"
    assert data["generation"]["temperature"] == 0.5


# --------------------------------------------------------------------------- #
# ingest command
# --------------------------------------------------------------------------- #


def test_ingest_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "sample.md"
    doc_path.write_text("Important content about authentication. " * 20, encoding="utf-8")

    result = runner.invoke(app, ["ingest", str(doc_path)])
    assert result.exit_code == 0, result.stdout
    assert "Ingested" in result.stdout
    assert doc_path.name in result.stdout or str(doc_path) in result.stdout


# --------------------------------------------------------------------------- #
# ask command
# --------------------------------------------------------------------------- #


def test_ask_command(tmp_path: Path) -> None:
    doc_path = tmp_path / "data.md"
    doc_path.write_text("Machine learning is a subset of AI. " * 20, encoding="utf-8")

    # Ingest first (separate invocation; mock uses in-memory store)
    ingest_result = runner.invoke(app, ["ingest", str(doc_path)])
    assert ingest_result.exit_code == 0

    # Ask (fresh services; mock stub still returns a deterministic answer)
    result = runner.invoke(app, ["ask", "what is machine learning"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip(), "ask must produce output"


def test_ask_command_stream() -> None:
    result = runner.invoke(app, ["ask", "hello", "--stream"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip()


def test_ask_command_no_config() -> None:
    """Ask without --config uses mock defaults."""
    result = runner.invoke(app, ["ask", "explain something"])
    assert result.exit_code == 0, result.stdout
    assert result.stdout.strip()


# --------------------------------------------------------------------------- #
# Help
# --------------------------------------------------------------------------- #


def test_app_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout.lower()
    assert "ask" in result.stdout.lower()
    assert "serve" in result.stdout.lower()
    assert "config" in result.stdout.lower()
