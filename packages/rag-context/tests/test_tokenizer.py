"""Tests for :mod:`rag_context.tokenizer`."""

from __future__ import annotations

from rag_context import WhitespaceCounter
from rag_context.tokenizer import ContextTokenizer
from tokenizers import Tokenizer


def test_whitespace_counter_counts_words(whitespace_tokenizer: WhitespaceCounter) -> None:
    ct = ContextTokenizer(whitespace_tokenizer)
    assert ct.count("hello world") == 2
    assert ct.count("one two three") == 3
    assert ct.count("") == 0
    assert ct.count("   ") == 0


def test_whitespace_truncate_to_max_tokens(whitespace_tokenizer: WhitespaceCounter) -> None:
    ct = ContextTokenizer(whitespace_tokenizer)
    text = "one two three four five"
    assert ct.count(text) == 5
    truncated = ct.truncate_to_tokens(text, 3)
    assert truncated == "one two three"
    assert ct.count(truncated) == 3
    # Already within budget is a no-op.
    assert ct.truncate_to_tokens("a b c", 5) == "a b c"


def test_hf_tokenizer_count(wordlevel_tokenizer: Tokenizer) -> None:
    ct = ContextTokenizer(wordlevel_tokenizer)
    assert ct.count("hello world foo bar baz") == 5


def test_hf_tokenizer_truncate(wordlevel_tokenizer: Tokenizer) -> None:
    ct = ContextTokenizer(wordlevel_tokenizer)
    text = "hello world foo bar baz"
    assert ct.count(text) == 5
    truncated = ct.truncate_to_tokens(text, 3)
    assert ct.count(truncated) == 3
    assert truncated == "hello world foo"
