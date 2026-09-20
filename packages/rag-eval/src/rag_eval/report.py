"""Human-readable evaluation report formatting."""

from __future__ import annotations

import math

from rag_core.evaluation import EvaluationResult

__all__ = ["format_report"]


def _fmt(value: float) -> str:
    """Format a float to 4 decimal places, guarding against NaN/Inf."""
    if math.isnan(value) or math.isinf(value):
        return "N/A"
    return f"{value:.4f}"


def format_report(result: EvaluationResult) -> str:
    """Render an :class:`EvaluationResult` as a markdown report.

    Includes run metadata, per-metric values (4 decimal places), failure
    counts, and case totals.
    """
    lines: list[str] = []

    # --- header ---
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Dataset | {result.dataset} |")
    lines.append(f"| Run ID | {result.run_id} |")
    config = result.config_hash or "N/A"
    lines.append(f"| Config Hash | {config} |")
    lines.append("")

    # --- failure summary ---
    failure_counts: dict[str, int] = {}
    for qe in result.per_query:
        if qe.failure_class is not None and qe.failure_class != "":
            failure_counts[qe.failure_class] = failure_counts.get(qe.failure_class, 0) + 1

    total = len(result.per_query)
    total_failures = sum(failure_counts.values())
    successful = total - total_failures

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total cases | {total} |")
    lines.append(f"| Successful | {successful} |")
    lines.append(f"| Failed | {total_failures} |")
    lines.append("")

    if failure_counts:
        lines.append("### Failures by class")
        lines.append("")
        lines.append("| Failure Class | Count |")
        lines.append("|---------------|-------|")
        for cls, count in sorted(failure_counts.items()):
            lines.append(f"| {cls} | {count} |")
        lines.append("")

    # --- metrics ---
    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for mr in result.metrics:
        val_str = _fmt(mr.value)
        k_suffix = f" (k={mr.k})" if mr.k is not None else ""
        lines.append(f"| {mr.name}{k_suffix} | {val_str} |")
    lines.append("")

    # --- model versions ---
    if result.model_versions:
        lines.append("## Model Versions")
        lines.append("")
        for name, version in sorted(result.model_versions.items()):
            lines.append(f"- {name}: {version}")
        lines.append("")

    return "\n".join(lines)
