"""Confidence aggregation helpers shared by engines and the router."""

from __future__ import annotations

from rag_ocr.engines.base import RawLine


def aggregate_confidence(lines: list[RawLine]) -> float | None:
    """Mean confidence across ``lines``; ``None`` when the list is empty."""
    if not lines:
        return None
    total = 0.0
    for line in lines:
        total += line.confidence
    return total / len(lines)
