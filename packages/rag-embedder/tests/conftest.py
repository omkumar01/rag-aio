"""Shared test fixtures for rag-embedder."""

from __future__ import annotations

import pytest
from rag_core.documents import Document, DocumentPage
from rag_embedder.tokenization import HFTokenizer, SimpleTokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


@pytest.fixture
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer()


@pytest.fixture
def small_doc() -> Document:
    """A single-page document with a moderate amount of text."""
    text = (
        "This is the first page of the document. " * 3
        + "It has multiple sentences here. "
        + "Some more text to fill the buffer. " * 2
        + "The end of the first page."
    )
    return Document(source_uri="test://small", text=text, id="doc-small")


@pytest.fixture
def multi_page_doc() -> Document:
    """A multi-page document where each page has text."""
    page1_text = "This is the first page. It contains some text about topic A. "
    page2_text = "This is the second page. It discusses topic B in detail. "
    full = page1_text + page2_text
    return Document(
        source_uri="test://multi",
        text=full,
        pages=[
            DocumentPage(page_number=1, text=page1_text),
            DocumentPage(page_number=2, text=page2_text),
        ],
        id="doc-multi",
    )


@pytest.fixture
def hf_tokenizer() -> HFTokenizer:
    """HFTokenizer wrapping a locally-built tokenizers.Tokenizer (no network)."""
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "this": 4,
        "is": 5,
        "a": 6,
        "test": 7,
        "hello": 8,
        "world": 9,
        "foo": 10,
        "bar": 11,
    }
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return HFTokenizer(tokenizer=tok, model_name_or_path="local-test")


class FakeStore:
    """Minimal async vector store complying with the rag-core VectorStore
    protocol plus ``upsert_sparse`` (SparseCapableStore)."""

    def __init__(self) -> None:
        self.points: dict[str, tuple[list[float], dict, str | None]] = {}
        self.sparse_points: dict[str, tuple[list[int], list[float], str | None]] = {}

    async def upsert(self, chunk_id, vector, payload, namespace=None):
        self.points[chunk_id] = (vector[:], dict(payload), namespace)

    async def upsert_sparse(self, chunk_id, indices, values, namespace=None):
        self.sparse_points[chunk_id] = (list(indices), list(values), namespace)

    async def search(self, vector, top_k, filters=None, namespace=None):
        return []

    async def delete(self, chunk_ids, namespace=None):
        for cid in chunk_ids:
            self.points.pop(cid, None)
            self.sparse_points.pop(cid, None)

    async def health(self):
        return True


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_store_cls() -> type[FakeStore]:
    """The class itself, for tests that construct stores with custom state."""
    return FakeStore


@pytest.fixture
def long_text() -> str:
    """Text long enough to be split into multiple chunks."""
    return (
        "This is a test document. " * 20
        + "It contains multiple sentences. " * 15
        + "The content is varied and repetitive. " * 10
        + "The final sentence concludes the document."
    )
