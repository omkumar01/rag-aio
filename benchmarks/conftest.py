"""Pytest fixtures for the rag-aio benchmark suite.

Holds a session-scoped :class:`~benchmarks.harness.ReportWriter` that scenario
tests append their results to; at session finish the writer flushes the shared
report into ``benchmarks/results/`` (created on demand, git-ignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.harness import ReportWriter

RESULTS_DIR = Path(__file__).parent / "results"

BENCHMARK_CONFIG: dict[str, object] = {
    "suite": "rag-aio-benchmarks",
    "workload": "small synthetic text corpus (30 docs / 200 chunks, mock embedders)",
    "notes": "tracemalloc tracks Python allocations only; generation tests require LM Studio",
}

_writer: ReportWriter | None = None


@pytest.fixture(scope="session")
def report_writer() -> ReportWriter:
    """Session-scoped results collector flushed to benchmarks/results/ at session end."""
    global _writer
    if _writer is None:
        _writer = ReportWriter(RESULTS_DIR, config=BENCHMARK_CONFIG)
    return _writer


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Flush the shared benchmark report once per pytest session."""
    if _writer is None or not _writer.results:
        return
    json_path, md_path = _writer.flush("rag-aio-benchmarks")
    print(f"\nbenchmark report written: {json_path}")
    print(f"benchmark report written: {md_path}")
