"""Small pytest-based benchmark harness for rag-aio.

Deliberately NOT pytest-benchmark: this module is pure standard library and
provides exactly what docs/performance/README.md promises — mean/median/P50/
P90/P95/P99, a separate cold (first) run, warm statistics, and a peak-RAM
figure via :mod:`tracemalloc`.

RAM caveat: ``tracemalloc`` tracks **Python-level allocations only**. Memory
allocated by native extensions (ONNX runtime, qdrant-client native mode, BM25
index buffers inside C extensions, model weights, ...) is invisible to it, so
``peak_ram_bytes`` is a lower bound on real process memory.

The primary entry point is pytest (``uv run pytest benchmarks -m benchmark``);
:class:`ReportWriter` is flushed from ``benchmarks/conftest.py`` at session end.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import time
import tracemalloc
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "ReportWriter",
    "percentile",
    "run_benchmark",
]


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of *values* for ``pct`` in ``[0, 100]``.

    Matches the classic ``(n - 1) * p`` rank definition (the same one
    ``statistics.quantiles(method="inclusive")`` and numpy's default use), so
    P50 equals the median and P0/P100 are the min/max.
    """
    if not values:
        raise ValueError("percentile() of an empty sequence")
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be within [0, 100], got {pct}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


@dataclass(frozen=True)
class BenchmarkCase:
    """Description of one benchmark scenario (used in reports)."""

    name: str
    description: str


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured statistics for one scenario.

    ``cold_ms`` is the very first (unwarmed) invocation. ``timings_ms`` holds
    only the *measured* iterations (warmups discarded). ``peak_ram_bytes`` is
    the tracemalloc peak across all runs — Python allocations only (see the
    module docstring).
    """

    name: str
    description: str
    iterations: int
    warmups: int
    cold_ms: float
    timings_ms: list[float]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    peak_ram_bytes: int


def _summarize(
    name: str,
    description: str,
    cold_ms: float,
    timings_ms: list[float],
    warmups: int,
    peak_ram_bytes: int,
) -> BenchmarkResult:
    timings = list(timings_ms)
    return BenchmarkResult(
        name=name,
        description=description,
        iterations=len(timings),
        warmups=warmups,
        cold_ms=cold_ms,
        timings_ms=timings,
        mean_ms=statistics.fmean(timings),
        median_ms=statistics.median(timings),
        min_ms=min(timings),
        max_ms=max(timings),
        stdev_ms=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        p50_ms=percentile(timings, 50.0),
        p90_ms=percentile(timings, 90.0),
        p95_ms=percentile(timings, 95.0),
        p99_ms=percentile(timings, 99.0),
        peak_ram_bytes=peak_ram_bytes,
    )


async def run_benchmark(
    name: str,
    fn: Callable[[], Awaitable[object] | object],
    *,
    iterations: int,
    warmup: int = 1,
    description: str = "",
) -> BenchmarkResult:
    """Run *fn* once cold, *warmup* discarded times, then *iterations* timed runs.

    ``fn`` is a zero-argument callable whose result may be awaitable; it is
    invoked exactly ``1 + warmup + iterations`` times. Timings are wall-clock
    milliseconds via :func:`time.perf_counter`. Peak traced memory covers the
    whole run (cold + warmups + measured).
    """
    if iterations < 1:
        raise ValueError(f"iterations must be >= 1, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    tracemalloc.start()
    try:
        # Cold (first, unwarmed) run — includes any lazy initialization cost.
        start = time.perf_counter()
        outcome = fn()
        if isinstance(outcome, Awaitable):
            await outcome
        cold_ms = (time.perf_counter() - start) * 1000.0

        for _ in range(warmup):
            outcome = fn()
            if isinstance(outcome, Awaitable):
                await outcome

        timings: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            outcome = fn()
            if isinstance(outcome, Awaitable):
                await outcome
            timings.append((time.perf_counter() - start) * 1000.0)

        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    return _summarize(name, description, cold_ms, timings, warmup, peak)


@dataclass
class ReportWriter:
    """Collects :class:`BenchmarkResult` objects and writes JSON + Markdown reports.

    Reports land in ``<results_dir>/<UTC timestamp>-<report_name>.json/.md``.
    The environment header records the Python version, platform, CPU count,
    report timestamp, and a short hash of *config* so runs stay reproducible
    and comparable.
    """

    results_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    _results: list[BenchmarkResult] = field(default_factory=list)

    @property
    def results(self) -> list[BenchmarkResult]:
        return list(self._results)

    def add(self, result: BenchmarkResult) -> None:
        self._results.append(result)

    def environment(self) -> dict[str, str]:
        canonical = json.dumps(self.config, sort_keys=True, default=str)
        config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": str(os.cpu_count() or 0),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "config_hash": config_hash,
        }

    def flush(self, report_name: str) -> tuple[Path, Path]:
        """Write the JSON (full data) and Markdown (table) reports; returns both paths."""
        if not self._results:
            raise ValueError("no results collected; nothing to flush")
        slug = report_name.strip().lower().replace(" ", "-") or "report"
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        ordered = sorted(self._results, key=lambda r: r.name)
        payload: dict[str, Any] = {
            "environment": self.environment(),
            "config": self.config,
            "results": [asdict(r) for r in ordered],
        }
        json_path = self.results_dir / f"{stamp}-{slug}.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        md_path = self.results_dir / f"{stamp}-{slug}.md"
        md_path.write_text(self._render_markdown(ordered), encoding="utf-8")
        return json_path, md_path

    def _render_markdown(self, results: list[BenchmarkResult]) -> str:
        env = self.environment()
        lines = [
            "# rag-aio benchmark report",
            "",
            f"- Python: {env['python_version']}",
            f"- Platform: {env['platform']}",
            f"- CPUs: {env['cpu_count']}",
            f"- Timestamp (UTC): {env['timestamp_utc']}",
            f"- Config hash: `{env['config_hash']}`",
            "",
            "Timings are warm-run milliseconds over the measured iterations; "
            "`cold` is the first unwarmed invocation. `peak RAM` is the "
            "tracemalloc peak (Python allocations only — native/model memory is "
            "not tracked).",
            "",
            "| benchmark | iterations | cold ms | mean | median | P50 | P90 | P95 | P99 | peak RAM (MiB) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in results:
            lines.append(
                f"| {r.name} | {r.iterations} | {r.cold_ms:.3f} | {r.mean_ms:.3f} | "
                f"{r.median_ms:.3f} | {r.p50_ms:.3f} | {r.p90_ms:.3f} | {r.p95_ms:.3f} | "
                f"{r.p99_ms:.3f} | {r.peak_ram_bytes / (1024 * 1024):.2f} |"
            )
        lines.append("")
        lines.append("## Scenario descriptions")
        lines.append("")
        for r in results:
            if r.description:
                lines.append(f"- **{r.name}**: {r.description}")
        lines.append("")
        return "\n".join(lines)
