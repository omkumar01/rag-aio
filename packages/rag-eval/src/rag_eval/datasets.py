"""Dataset models and loaders for RAG evaluation cases."""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError
from rag_core.base import RagBaseModel
from rag_core.errors import EvaluationError

__all__ = [
    "EvalCase",
    "load_csv",
    "load_jsonl",
    "load_parquet",
    "save_jsonl",
]

# --- parquet import guard --------------------------------------------------

try:
    import pyarrow.parquet as _pq  # type: ignore[import-untyped]

    _HAS_PYARROW: bool = True
except ImportError:  # pragma: no cover
    _HAS_PYARROW = False
    _pq = None


class EvalCase(RagBaseModel):
    """A single evaluation case.

    ``relevant_ids`` are the ground-truth document/chunk ids that should be
    retrieved for ``query``.  ``golden_answer`` is the reference answer
    (optional; required for answer-generation metrics).
    """

    query_id: str
    query: str
    relevant_ids: list[str]
    golden_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# --- JSONL ----------------------------------------------------------------


def _safe_dataset_path(path: str | Path) -> Path:
    """Validate and resolve a dataset path.

    Rejects ``..`` traversal segments so callers can never address files
    outside the intended directory through relative tricks.
    """
    p = Path(path).expanduser()
    if ".." in p.parts:
        raise EvaluationError(
            f"dataset path must not contain '..' segments: {path}",
            code="unsafe_path",
        )
    return p.resolve()


def load_jsonl(path: str | Path) -> list[EvalCase]:
    """Load evaluation cases from a JSONL file.

    Each line must be a JSON object decodable into :class:`EvalCase`.
    Blank lines are skipped.  Raises :class:`EvaluationError` with the
    offending line number on any parse or validation failure.
    """
    cases: list[EvalCase] = []
    with open(_safe_dataset_path(path), encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"Invalid JSON on line {line_no}",
                    code="invalid_json",
                    details={"line": line_no, "error": str(exc)},
                ) from exc
            try:
                cases.append(EvalCase.model_validate(data))
            except ValidationError as exc:
                raise EvaluationError(
                    f"Invalid case on line {line_no}",
                    code="invalid_case",
                    details={"line": line_no, "error": exc.errors()},
                ) from exc
    return cases


def save_jsonl(cases: list[EvalCase], path: str | Path, base_dir: str | Path) -> None:
    """Persist evaluation cases to a JSONL file (one JSON object per line).

    ``base_dir`` is required: only the final filename component of ``path``
    is used, joined onto ``base_dir``, so the write target can never escape
    the base directory by construction.
    """
    root = Path(base_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / Path(path).name
    payload = "".join(case.model_dump_json() + "\n" for case in cases)
    target.write_text(payload, encoding="utf-8")


# --- CSV ------------------------------------------------------------------


def load_csv(path: str | Path) -> list[EvalCase]:
    """Load cases from a CSV file.

    Expected columns: ``query_id``, ``query``, ``relevant_ids``
    (``;``-separated), and optionally ``golden_answer``.
    """
    cases: list[EvalCase] = []
    with open(_safe_dataset_path(path), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_no, row in enumerate(reader, 2):  # header is line 1
            query_id = row.get("query_id", "")
            query = row.get("query", "")
            if not query_id or not query:
                raise EvaluationError(
                    f"Missing query_id or query on row {row_no}",
                    code="invalid_case",
                    details={"row": row_no},
                )
            raw_ids = row.get("relevant_ids", "")
            relevant = [r.strip() for r in raw_ids.split(";") if r.strip()]
            golden = row.get("golden_answer") or None
            try:
                cases.append(
                    EvalCase(
                        query_id=query_id,
                        query=query,
                        relevant_ids=relevant,
                        golden_answer=golden,
                    )
                )
            except ValidationError as exc:
                raise EvaluationError(
                    f"Invalid case on row {row_no}",
                    code="invalid_case",
                    details={"row": row_no, "error": exc.errors()},
                ) from exc
    return cases


# --- Parquet (optional) ---------------------------------------------------


def load_parquet(path: str | Path) -> list[EvalCase]:
    """Load cases from a Parquet file (requires ``pyarrow``)."""
    if not _HAS_PYARROW:
        raise EvaluationError(
            "Parquet support requires pyarrow. Install with: pip install pyarrow",
            code="parquet_not_installed",
        )
    assert _pq is not None  # narrowed by _HAS_PYARROW check
    table = _pq.read_table(str(path))
    records: list[dict[str, Any]] = table.to_pylist()
    cases: list[EvalCase] = []
    for row_no, record in enumerate(records, 1):
        if isinstance(record.get("relevant_ids"), str):
            record = copy.copy(record)
            record["relevant_ids"] = [
                r.strip() for r in record["relevant_ids"].split(";") if r.strip()
            ]
        try:
            cases.append(EvalCase.model_validate(record))
        except ValidationError as exc:
            raise EvaluationError(
                f"Invalid case on row {row_no}",
                code="invalid_case",
                details={"row": row_no, "error": exc.errors()},
            ) from exc
    return cases
