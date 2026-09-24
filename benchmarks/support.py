"""Shared workload generators and offline-safety helpers for the scenario tests."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

from benchmarks.harness import BenchmarkResult

__all__ = [
    "check_sane",
    "fastembed_cache_dir",
    "fastembed_model_cached",
    "generate_chunk_text",
    "generate_doc_text",
]

# A fixed word bank keeps every generated document deterministic across runs
# and across machines, so timings stay comparable between report files.
_WORDS: tuple[str, ...] = (
    "retrieval",
    "augmented",
    "generation",
    "pipeline",
    "embedding",
    "chunking",
    "index",
    "latency",
    "throughput",
    "orchestrator",
    "citation",
    "context",
    "token",
    "budget",
    "sparse",
    "dense",
    "hybrid",
    "reranker",
    "document",
    "parser",
    "ingestion",
    "namespace",
    "vector",
    "store",
    "mock",
    "benchmark",
    "cache",
    "query",
    "fusion",
    "percentile",
)


def generate_doc_text(index: int, *, paragraphs: int = 10, words_per_paragraph: int = 50) -> str:
    """Deterministic synthetic document (~2-3 KB) with a heading and body paragraphs."""
    parts = [f"# Document {index}: sample topic {index % 5}"]
    offset = index * 7
    for p in range(paragraphs):
        words = [_WORDS[(offset + p * 3 + i) % len(_WORDS)] for i in range(words_per_paragraph)]
        parts.append(" ".join(words))
    return "\n\n".join(parts)


def generate_chunk_text(index: int, *, words: int = 40) -> str:
    """Deterministic synthetic chunk text (one short paragraph)."""
    offset = index * 11
    picked = [_WORDS[(offset + i) % len(_WORDS)] for i in range(words)]
    return f"Chunk {index}: " + " ".join(picked)


def fastembed_cache_dir() -> Path | None:
    """The fastembed download cache directory, or ``None`` if it does not exist.

    Mirrors fastembed's own resolution: ``$FASTEMBED_CACHE_PATH`` if set,
    otherwise ``<system temp dir>/fastembed_cache``.
    """
    cache = Path(
        os.getenv("FASTEMBED_CACHE_PATH", os.path.join(tempfile.gettempdir(), "fastembed_cache"))
    )
    return cache if cache.is_dir() else None


def fastembed_model_cached(model_name: str) -> bool:
    """Best-effort check that *model_name* is already downloaded locally.

    fastembed stores HuggingFace-style snapshot directories named
    ``models--<org>--<model>`` inside its cache dir; we look for the model's
    last path segment case-insensitively so renamed variants (e.g.
    ``models--qdrant--bge-small-en-v1.5-onnx-q`` for ``BAAI/bge-small-en-v1.5``)
    still match. This keeps the fastembed scenario offline-safe: if the model
    is not already on disk we skip rather than trigger a download.
    """
    cache = fastembed_cache_dir()
    if cache is None:
        return False
    needle = model_name.rsplit("/", 1)[-1].lower()
    return any(needle in entry.name.lower() for entry in cache.iterdir())


def check_sane(result: BenchmarkResult, *, expected_iterations: int) -> None:
    """Assert the recorded result has the expected shape and finite statistics."""
    assert result.iterations == expected_iterations
    assert len(result.timings_ms) == expected_iterations
    assert result.cold_ms >= 0.0
    fields = ("mean_ms", "median_ms", "min_ms", "max_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms")
    for name in fields:
        value = getattr(result, name)
        assert math.isfinite(value), f"{name} is not finite: {value}"
    assert all(math.isfinite(t) and t >= 0.0 for t in result.timings_ms)
    assert result.peak_ram_bytes >= 0
