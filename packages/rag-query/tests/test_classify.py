"""Tests for classify_query."""

from __future__ import annotations

import pytest
from rag_query.classify import QueryClass, classify_query


@pytest.mark.parametrize(
    "text, expected",
    [
        ("LLM retrieval", QueryClass.keyword),
        ("foo bar", QueryClass.keyword),
        ("vector db", QueryClass.keyword),
        ("what is RAG", QueryClass.question),
        ("how does retrieval work", QueryClass.question),
        ("why is the sky blue", QueryClass.question),
        ("tell me?", QueryClass.question),
        ("show me the best papers", QueryClass.navigational),
        ("find me recent research", QueryClass.navigational),
        ("SHOW ME the money", QueryClass.navigational),
        (
            "Tell me something about retrieval in RAG systems",
            QueryClass.conversational,
        ),
        ("I need to understand dense retrieval pipelines", QueryClass.conversational),
    ],
)
def test_classify(text: str, expected: QueryClass) -> None:
    assert classify_query(text) == expected


def test_returns_str_enum() -> None:
    result = classify_query("what is RAG")
    assert result.value == "question"
    assert isinstance(result.value, str)
