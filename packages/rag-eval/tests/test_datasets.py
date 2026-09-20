"""Tests for dataset loaders and EvalCase."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from rag_core.errors import EvaluationError
from rag_eval.datasets import (
    _HAS_PYARROW,
    EvalCase,
    load_csv,
    load_jsonl,
    load_parquet,
    save_jsonl,
)

pytestmark = pytest.mark.unit


# --- EvalCase construction --------------------------------------------------


def test_eval_case_basic() -> None:
    case = EvalCase(
        query_id="q1",
        query="what is RAG?",
        relevant_ids=["doc1", "doc2"],
        golden_answer="Retrieval-Augmented Generation",
    )
    assert case.query_id == "q1"
    assert case.relevant_ids == ["doc1", "doc2"]
    assert case.golden_answer == "Retrieval-Augmented Generation"
    assert case.metadata == {}


def test_eval_case_defaults() -> None:
    case = EvalCase(query_id="q1", query="q", relevant_ids=[])
    assert case.golden_answer is None
    assert case.metadata == {}


def test_eval_case_extra_field_forbidden() -> None:
    with pytest.raises(ValidationError):
        EvalCase(query_id="q1", query="q", relevant_ids=["a"], extra="bad")


# --- JSONL roundtrip --------------------------------------------------------


def test_jsonl_roundtrip(tmp_path) -> None:
    cases = [
        EvalCase(
            query_id="q1",
            query="what?",
            relevant_ids=["a", "b"],
            golden_answer="ans1",
            metadata={"source": "test"},
        ),
        EvalCase(
            query_id="q2",
            query="why?",
            relevant_ids=["c"],
        ),
    ]
    path = tmp_path / "cases.jsonl"
    save_jsonl(cases, path, base_dir=tmp_path)
    loaded = load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0].query_id == "q1"
    assert loaded[0].relevant_ids == ["a", "b"]
    assert loaded[0].golden_answer == "ans1"
    assert loaded[0].metadata == {"source": "test"}
    assert loaded[1].query_id == "q2"
    assert loaded[1].golden_answer is None


def test_save_jsonl_confines_to_base_dir(tmp_path) -> None:
    # A traversal segment is reduced to its final filename and written
    # strictly inside base_dir, so the parent directory is never touched.
    cases = [EvalCase(query_id="q1", query="q", relevant_ids=["a"])]
    parent = tmp_path / "parent"
    parent.mkdir()
    base = parent / "store"
    base.mkdir()
    save_jsonl(cases, "../evil.jsonl", base_dir=base)
    assert (base / "evil.jsonl").exists()
    assert not (parent / "evil.jsonl").exists()


def test_jsonl_blank_lines_skipped(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"query_id": "q1", "query": "q", "relevant_ids": ["a"]}\n'
        "\n"
        '{"query_id": "q2", "query": "q2", "relevant_ids": ["b"]}\n'
    )
    loaded = load_jsonl(path)
    assert len(loaded) == 2


def test_jsonl_bad_json_raises_with_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"query_id": "q1", "query": "q", "relevant_ids": ["a"]}\nnot valid json\n')
    with pytest.raises(EvaluationError, match="line 2") as exc_info:
        load_jsonl(path)
    assert "line 2" in str(exc_info.value)


def test_jsonl_bad_case_raises_with_line_number(tmp_path) -> None:
    path = tmp_path / "bad2.jsonl"
    path.write_text('{"query_id": "q1", "query": "q", "relevant_ids": ["a"]\n')
    with pytest.raises(EvaluationError, match="line 1"):
        load_jsonl(path)


def test_jsonl_extra_field_rejected(tmp_path) -> None:
    path = tmp_path / "extra.jsonl"
    path.write_text('{"query_id": "q1", "query": "q", "relevant_ids": ["a"], "extra": "x"}\n')
    with pytest.raises(EvaluationError, match="Invalid case"):
        load_jsonl(path)


# --- CSV --------------------------------------------------------------------


def test_csv_basic(tmp_path) -> None:
    path = tmp_path / "cases.csv"
    path.write_text(
        "query_id,query,relevant_ids,golden_answer\n"
        "q1,what is RAG?,a;b;c,Retrieval Augmented Generation\n"
        'q2,how it works,d,""\n'
    )
    loaded = load_csv(path)
    assert len(loaded) == 2
    assert loaded[0].query_id == "q1"
    assert loaded[0].relevant_ids == ["a", "b", "c"]
    assert loaded[0].golden_answer == "Retrieval Augmented Generation"
    assert loaded[1].query_id == "q2"
    assert loaded[1].relevant_ids == ["d"]
    assert loaded[1].golden_answer is None


def test_csv_with_spaces_in_ids(tmp_path) -> None:
    path = tmp_path / "cases.csv"
    path.write_text("query_id,query,relevant_ids\nq1,what?, a ; b ; c\n")
    loaded = load_csv(path)
    assert loaded[0].relevant_ids == ["a", "b", "c"]


# --- Parquet ----------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PYARROW, reason="pyarrow not installed")
def test_parquet_roundtrip(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = [
        {
            "query_id": "q1",
            "query": "what?",
            "relevant_ids": ["a", "b"],
            "golden_answer": "ans1",
        },
        {
            "query_id": "q2",
            "query": "why?",
            "relevant_ids": ["c"],
            "golden_answer": None,
        },
    ]
    table = pa.Table.from_pylist(records)
    path = tmp_path / "cases.parquet"
    pq.write_table(table, str(path))
    loaded = load_parquet(path)
    assert len(loaded) == 2
    assert loaded[0].query_id == "q1"
    assert loaded[0].relevant_ids == ["a", "b"]
    assert loaded[0].golden_answer == "ans1"
    assert loaded[1].golden_answer is None


def test_load_parquet_missing_pyarrow(monkeypatch) -> None:
    """When pyarrow is unavailable, raise EvaluationError with install hint."""
    import rag_eval.datasets as datasets_mod

    monkeypatch.setattr(datasets_mod, "_HAS_PYARROW", False)
    with pytest.raises(EvaluationError, match="pyarrow"):
        datasets_mod.load_parquet("dummy.parquet")
