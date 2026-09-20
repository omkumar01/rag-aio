"""Tests for generation quality metrics and LLM judge."""

from __future__ import annotations

import pytest
from rag_core.generation import GenerationRequest, GenerationResult
from rag_eval.generation_metrics import (
    OpenAICompatibleJudge,
    answer_overlap,
    citation_precision,
    citation_recall,
    faithfulness_proxy,
)

# --- citation metrics ------------------------------------------------------


def test_citation_precision_all_correct() -> None:
    assert citation_precision(["a", "b"], {"a", "b"}) == 1.0


def test_citation_precision_half() -> None:
    assert citation_precision(["a", "b", "c"], {"a", "b"}) == pytest.approx(2 / 3)


def test_citation_precision_empty_pred() -> None:
    assert citation_precision([], {"a"}) == 0.0


def test_citation_recall_all() -> None:
    assert citation_recall({"a", "b"}, ["a", "b"]) == 1.0


def test_citation_recall_partial() -> None:
    assert citation_recall({"a", "b", "c"}, ["a", "b"]) == pytest.approx(2 / 3)


def test_citation_recall_empty_supported() -> None:
    assert citation_recall(set(), ["a"]) == 0.0


# --- answer overlap --------------------------------------------------------


def test_answer_overlap_identical() -> None:
    assert answer_overlap("hello world", ["hello world foo bar"]) == pytest.approx(0.5)


def test_answer_overlap_no_context() -> None:
    assert answer_overlap("hello world", []) == 0.0


def test_answer_overlap_empty_answer() -> None:
    assert answer_overlap("", ["hello world"]) == 0.0


def test_answer_overlap_disjoint() -> None:
    assert answer_overlap("foo bar", ["baz qux"]) == 0.0


def test_answer_overlap_subset() -> None:
    # answer tokens all in context -> intersection == answer set
    answer = "hello world"
    context = "hello world foo bar"
    ans_tokens = {"hello", "world"}
    ctx_tokens = {"hello", "world", "foo", "bar"}
    expected = len(ans_tokens & ctx_tokens) / len(ans_tokens | ctx_tokens)
    assert answer_overlap(answer, [context]) == pytest.approx(expected)


# --- faithfulness proxy ----------------------------------------------------


def test_faithfulness_full_coverage() -> None:
    answer = "the cat sat on the mat"
    context = "the cat sat on the mat and the dog"
    # all 4-gram shingles of answer appear in context
    assert faithfulness_proxy(answer, [context]) == 1.0


def test_faithfulness_no_context() -> None:
    assert faithfulness_proxy("hello world test data", []) == 0.0


def test_faithfulness_partial() -> None:
    answer = "the cat sat on the mat"
    context = "the dog sat on the mat"
    # answer shingles: {the,cat,sat,on}, {cat,sat,on,the}, {sat,on,the,mat}
    # context shingles: {the,dog,sat,on}, {dog,sat,on,the}, {sat,on,the,mat}
    # only {sat,on,the,mat} matches -> 1/3
    assert faithfulness_proxy(answer, [context]) == pytest.approx(1 / 3)


def test_faithfulness_no_shingles() -> None:
    # fewer than 4 tokens -> no 4-grams
    assert faithfulness_proxy("hi", ["some context text"]) == 0.0


# --- LLM judge -------------------------------------------------------------


class _FakeGenerator:
    """Minimal async generator for testing."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(
            text=self._response_text,
            model=request.model or "fake-model",
            finish_reason="stop",
        )


@pytest.mark.unit
async def test_llm_judge_valid_json() -> None:
    gen = _FakeGenerator('{"score": 0.85, "label": "faithful", "reasoning": "well supported"}')
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("The answer", "some context", "a question")
    assert verdict.score == pytest.approx(0.85)
    assert verdict.label == "faithful"
    assert verdict.reasoning == "well supported"


@pytest.mark.unit
async def test_llm_judge_json_in_markdown() -> None:
    gen = _FakeGenerator(
        '```json\n{"score": 0.3, "label": "unfaithful", "reasoning": "hallucinated"}\n```'
    )
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("bad answer", "context", "q")
    assert verdict.score == pytest.approx(0.3)
    assert verdict.label == "unfaithful"


@pytest.mark.unit
async def test_llm_judge_malformed() -> None:
    gen = _FakeGenerator("not json at all")
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("answer", "context", "question")
    assert verdict.score == 0.0
    assert verdict.label == "error"


@pytest.mark.unit
async def test_llm_judge_score_clamped() -> None:
    gen = _FakeGenerator('{"score": 1.5, "label": "faithful", "reasoning": "too high"}')
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("a", "b", "q")
    assert verdict.score == 1.0


@pytest.mark.unit
async def test_llm_judge_score_negative_clamped() -> None:
    gen = _FakeGenerator('{"score": -0.5, "label": "unfaithful", "reasoning": "bad"}')
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("a", "b", "q")
    assert verdict.score == 0.0


@pytest.mark.unit
async def test_llm_judge_reasoning_optional() -> None:
    gen = _FakeGenerator('{"score": 0.5, "label": "partial", "reasoning": null}')
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("a", "b", "q")
    assert verdict.score == pytest.approx(0.5)
    assert verdict.label == "partial"
    assert verdict.reasoning is None


@pytest.mark.unit
async def test_llm_judge_surrounding_text() -> None:
    gen = _FakeGenerator(
        'Here is my evaluation: {"score": 0.7, "label": "faithful", "reasoning": "good"} Done.'
    )
    judge = OpenAICompatibleJudge(generator=gen, model="test")
    verdict = await judge.judge("a", "b", "q")
    assert verdict.score == pytest.approx(0.7)
    assert verdict.label == "faithful"
