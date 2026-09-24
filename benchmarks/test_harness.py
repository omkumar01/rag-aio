"""Unit tests for the benchmark harness math (fast, deterministic, stdlib-only).

Deliberately imports only :mod:`benchmarks.harness` (pure stdlib) so this test
can run in fast CI under the ``unit`` marker without pulling in the rag-*
packages.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.harness import BenchmarkResult, percentile, run_benchmark

pytestmark = pytest.mark.unit


def test_percentile_single_value() -> None:
    assert percentile([42.0], 50.0) == 42.0
    assert percentile([42.0], 0.0) == 42.0
    assert percentile([42.0], 100.0) == 42.0


def test_percentile_linear_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    # rank = p / 100 * (n - 1); interpolate between neighbouring order stats.
    assert percentile(values, 0.0) == 10.0
    assert percentile(values, 25.0) == pytest.approx(17.5)
    assert percentile(values, 50.0) == pytest.approx(25.0)
    assert percentile(values, 90.0) == pytest.approx(37.0)
    assert percentile(values, 100.0) == 40.0


def test_percentile_matches_median_and_extremes() -> None:
    values = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert percentile(values, 50.0) == 3.0  # odd count: exact order statistic
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 100.0) == 5.0


def test_percentile_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50.0)
    with pytest.raises(ValueError, match="within"):
        percentile([1.0], 101.0)
    with pytest.raises(ValueError, match="within"):
        percentile([1.0], -1.0)


async def test_run_benchmark_counts_cold_warmup_and_measured() -> None:
    calls = 0

    def tiny() -> int:
        nonlocal calls
        calls += 1
        return sum(range(1000))

    result = await run_benchmark(
        "harness_selfcheck",
        tiny,
        iterations=5,
        warmup=2,
        description="harness unit check",
    )
    assert calls == 1 + 2 + 5
    assert result.iterations == 5
    assert result.warmups == 2
    assert len(result.timings_ms) == 5
    assert result.cold_ms >= 0.0
    assert all(math.isfinite(t) and t >= 0.0 for t in result.timings_ms)
    assert all(
        math.isfinite(getattr(result, f))
        for f in ("mean_ms", "median_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms")
    )
    assert result.peak_ram_bytes >= 0


def test_benchmark_result_synthetic_timings_are_consistent() -> None:
    result = BenchmarkResult(
        name="synthetic",
        description="",
        iterations=4,
        warmups=1,
        cold_ms=100.0,
        timings_ms=[10.0, 20.0, 30.0, 40.0],
        mean_ms=25.0,
        median_ms=25.0,
        min_ms=10.0,
        max_ms=40.0,
        stdev_ms=0.0,
        p50_ms=25.0,
        p90_ms=37.0,
        p95_ms=38.5,
        p99_ms=39.7,
        peak_ram_bytes=2048,
    )
    assert result.min_ms == min(result.timings_ms)
    assert result.max_ms == max(result.timings_ms)
    assert result.p50_ms == result.median_ms
