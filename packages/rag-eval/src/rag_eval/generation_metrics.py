"""Generation quality metrics: citation correctness and answer groundedness.

All heuristics here are LLM-free.  LLM-based judges are provided as adapters
that wrap a duck-typed generator, never as a default.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from rag_core.base import RagBaseModel
from rag_core.errors import EvaluationError
from rag_core.generation import GenerationRequest, GenerationResult, Message

__all__ = [
    "JudgeVerdict",
    "LLMJudge",
    "OpenAICompatibleJudge",
    "answer_overlap",
    "citation_precision",
    "citation_recall",
    "faithfulness_proxy",
]

# --- tokenisation helpers -------------------------------------------------

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercased word-token extraction."""
    return _TOKEN_RE.findall(text.lower())


def _shingles(text: str, n: int) -> set[tuple[str, ...]]:
    """Set of *n*-gram token tuples from *text*."""
    tokens = _tokenize(text)
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# --- citation metrics -----------------------------------------------------


def citation_precision(predicted_citation_ids: list[str], supported_ids: set[str]) -> float:
    """Precision of citations: fraction of predicted citations that are supported.

    Parameters
    ----------
    predicted_citation_ids
        Citation ids produced by the pipeline.
    supported_ids
        Ground-truth citation ids that actually support the answer.
    """
    if not predicted_citation_ids:
        return 0.0
    predicted = set(predicted_citation_ids)
    if not predicted:
        return 0.0
    return len(predicted & supported_ids) / len(predicted)


def citation_recall(supported_ids: set[str], predicted_citation_ids: list[str]) -> float:
    """Recall of citations: fraction of supporting docs that were cited.

    Parameters
    ----------
    supported_ids
        Ground-truth citation ids that actually support the answer.
    predicted_citation_ids
        Citation ids produced by the pipeline.
    """
    if not supported_ids:
        return 0.0
    predicted = set(predicted_citation_ids)
    return len(predicted & supported_ids) / len(supported_ids)


def answer_overlap(answer: str, context_texts: list[str]) -> float:
    """Jaccard overlap of answer token set vs. context token set.

    A proxy for groundedness: an answer whose tokens all appear in the context
    scores 1.0; novel (hallucinated) tokens reduce the score.
    """
    answer_tokens = set(_tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for text in context_texts:
        context_tokens.update(_tokenize(text))
    if not context_tokens:
        return 0.0
    intersection = answer_tokens & context_tokens
    union = answer_tokens | context_tokens
    return len(intersection) / len(union)


def faithfulness_proxy(answer: str, context_texts: list[str]) -> float:
    """Fraction of 4-gram shingles from *answer* present in *context*.

    Returns 0.0 when there are no context texts or no answer shingles.
    """
    if not context_texts:
        return 0.0
    answer_shingles = _shingles(answer, 4)
    if not answer_shingles:
        return 0.0
    context_shingles: set[tuple[str, ...]] = set()
    for text in context_texts:
        context_shingles.update(_shingles(text, 4))
    if not context_shingles:
        return 0.0
    present = sum(1 for s in answer_shingles if s in context_shingles)
    return present / len(answer_shingles)


# --- LLM judge ------------------------------------------------------------


class JudgeVerdict(RagBaseModel):
    """Outcome of an LLM-as-judge evaluation."""

    score: float
    label: str
    reasoning: str | None = None


@runtime_checkable
class LLMJudge(Protocol):
    """Protocol for judge objects that evaluate answer faithfulness."""

    async def judge(self, answer: str, context: str, question: str) -> JudgeVerdict: ...


@runtime_checkable
class GeneratorLike(Protocol):
    """Duck-typed generator with a single ``generate`` coroutine."""

    async def generate(self, request: GenerationRequest) -> GenerationResult: ...


_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial evaluation judge. Evaluate the provided answer for "
    "faithfulness to the provided context. An answer is faithful if every claim "
    "is supported by the context, and unfaithful if it contains unsupported claims.\n\n"
    "Output ONLY valid JSON with these exact keys:\n"
    '{"score": <float 0.0-1.0>, "label": "<faithful|unfaithful|partial>", "reasoning": "<string or null>"}'
)

_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Context: {context}\n\n"
    "Answer: {answer}\n\n"
    "Evaluate and output JSON only."
)

_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "label": {"type": "string"},
        "reasoning": {"type": ["string", "null"]},
    },
    "required": ["score", "label", "reasoning"],
    "additionalProperties": False,
}


class OpenAICompatibleJudge:
    """LLM judge that wraps a duck-typed generator.

    Parameters
    ----------
    generator
        Any object with ``async def generate(request: GenerationRequest) ->
        GenerationResult``.
    model, temperature, max_tokens
        Passed through to the :class:`GenerationRequest`.
    """

    def __init__(
        self,
        generator: GeneratorLike,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = 4096,
    ) -> None:
        self._generator = generator
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def judge(self, answer: str, context: str, question: str) -> JudgeVerdict:
        messages = [
            Message(role="system", content=_JUDGE_SYSTEM_PROMPT),
            Message(
                role="user",
                content=_USER_TEMPLATE.format(question=question, context=context, answer=answer),
            ),
        ]
        request = GenerationRequest(
            messages=messages,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            json_schema=_JSON_SCHEMA,
        )
        result = await self._generator.generate(request)
        return _parse_judge_response(result.text)


def _parse_judge_response(text: str) -> JudgeVerdict:
    """Extract a :class:`JudgeVerdict` from raw LLM text.

    Falls back to ``score=0, label="error"`` on any parsing failure.
    """
    try:
        data = _extract_json(text)
    except EvaluationError:
        return JudgeVerdict(score=0.0, label="error", reasoning="Malformed LLM response")

    score_raw = data.get("score", 0.0)
    score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
    score = max(0.0, min(1.0, score))

    label_raw = data.get("label", "error")
    label = str(label_raw) if label_raw is not None else "error"

    reasoning_raw = data.get("reasoning")
    reasoning: str | None = str(reasoning_raw) if reasoning_raw is not None else None

    return JudgeVerdict(score=score, label=label, reasoning=reasoning)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Tries direct parse, then markdown code blocks, then brace matching.
    """
    cleaned = text.strip()

    # 1. Direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2. Markdown code block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # 3. First { ... last }
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    raise EvaluationError("Could not parse JSON from LLM judge response", code="judge_json_parse")
