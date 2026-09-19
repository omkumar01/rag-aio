"""Tests for tokenization.py — no network, no model downloads."""

from __future__ import annotations

from rag_embedder.tokenization import HFTokenizer, SimpleTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


class TestSimpleTokenizer:
    def test_count_tokens_positive(self):
        tok = SimpleTokenizer()
        assert tok.count_tokens("hello world foo bar") > 0

    def test_encode_returns_ids(self):
        tok = SimpleTokenizer()
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) == 2

    def test_decode_roundtrip(self):
        tok = SimpleTokenizer()
        text = "hello world this is a test"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # SimpleTokenizer joins tokens with spaces.
        assert decoded == text

    def test_count_tokens_matches_encode_length(self):
        tok = SimpleTokenizer()
        text = "the quick brown fox"
        assert tok.count_tokens(text) == len(tok.encode(text))

    def test_empty_text(self):
        tok = SimpleTokenizer()
        assert tok.count_tokens("") == 0
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_deterministic_within_instance(self):
        tok = SimpleTokenizer()
        text = "deterministic text here"
        ids1 = tok.encode(text)
        ids2 = tok.encode(text)
        assert ids1 == ids2


class TestHFTokenizer:
    def test_local_tokenizer_no_download(self, hf_tokenizer: HFTokenizer):
        assert hf_tokenizer.model_name == "local-test"

    def test_count_tokens(self, hf_tokenizer: HFTokenizer):
        assert hf_tokenizer.count_tokens("this is a test") > 0

    def test_encode_decode(self, hf_tokenizer: HFTokenizer):
        ids = hf_tokenizer.encode("hello world foo bar")
        assert isinstance(ids, list)
        decoded = hf_tokenizer.decode(ids)
        assert "hello" in decoded.lower()

    def test_from_tokenizer_classmethod(self):
        vocab = {"[UNK]": 0, "alpha": 1, "beta": 2}
        tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        hf = HFTokenizer.from_tokenizer(tok, model_name="tiny")
        assert hf.model_name == "tiny"
        assert hf.count_tokens("alpha beta gamma") == 3

    def test_fallback_when_no_tokenizer(self):
        """When from_pretrained fails, HFTokenizer falls back to SimpleTokenizer."""
        hf = HFTokenizer(model_name_or_path="nonexistent/model-xyz-12345")
        # Should not raise — falls back to SimpleTokenizer
        assert hf.count_tokens("hello world") > 0
        assert isinstance(hf.encode("hello world"), list)

    def test_tokenizer_protocol_methods(self, hf_tokenizer: HFTokenizer):
        """HFTokenizer satisfies the Tokenizer protocol."""
        tok = hf_tokenizer
        assert hasattr(tok, "count_tokens")
        assert hasattr(tok, "encode")
        assert hasattr(tok, "decode")
        assert tok.count_tokens("test") == len(tok.encode("test"))
