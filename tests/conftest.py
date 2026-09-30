"""Offline test doubles: no model downloads, no LiteParse CLI and no Groq calls."""

import hashlib
import math
import re
from types import SimpleNamespace

import pytest

from app.document_parser import Page
from app.llm_generator import LLMClient
from app.pipeline import RAGPipeline
from app.retrieval import HybridRetriever
from app.vector_db import VectorStore

_WORD = re.compile(r"\w+")


def count_words(text: str) -> int:
    return len(text.split())


class FakeEmbedder:
    """Hashing bag-of-words embedder: texts that share words get similar vectors."""

    max_tokens = 254

    def __init__(self, dim: int = 256):
        self.dim = dim

    def count_tokens(self, text: str) -> int:
        return count_words(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _WORD.findall(text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FakeParser:
    """Returns canned pages for known PDF bytes, or raises a canned error."""

    def __init__(self, documents: dict[bytes, list[Page] | Exception]):
        self.documents = documents
        self.calls = 0

    def parse(self, pdf_bytes: bytes) -> list[Page]:
        self.calls += 1
        result = self.documents[pdf_bytes]
        if isinstance(result, Exception):
            raise result
        return result


class FakeChatClient:
    """Mimics ``groq.Groq().chat.completions.create`` and records every call."""

    def __init__(self, replies: list[str] | None = None, error: Exception | None = None):
        self.calls: list[dict] = []
        self.replies = list(replies or [])
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.replies.pop(0) if self.replies else "Stub answer [1]."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


EIFFEL_PDF = b"%PDF-1.7 eiffel"
BIOLOGY_PDF = b"%PDF-1.7 biology"
PYTHON_PDF = b"%PDF-1.7 python"
BLANK_PDF = b"%PDF-1.7 blank"

DOCUMENTS: dict[bytes, list[Page] | Exception] = {
    EIFFEL_PDF: [
        Page(1, "The Eiffel Tower is located in Paris. It was completed in 1889 for the World's Fair."),
        Page(2, "The tower is 330 metres tall. Gustave Eiffel's company designed and built it."),
    ],
    BIOLOGY_PDF: [
        Page(
            1,
            "Photosynthesis converts light energy into chemical energy. "
            "Chlorophyll absorbs blue and red light.",
        ),
    ],
    PYTHON_PDF: [
        Page(1, "Python was created by Guido van Rossum. Python 3.0 was released in December 2008."),
    ],
    BLANK_PDF: [Page(1, "   "), Page(2, "\n\n")],
}


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder) -> VectorStore:
    return VectorStore(str(tmp_path / "chroma"), "test", embedder)


@pytest.fixture
def chat_client() -> FakeChatClient:
    return FakeChatClient()


@pytest.fixture
def make_pipeline(store, embedder, chat_client):
    def make(
        documents: dict[bytes, list[Page] | Exception] = DOCUMENTS,
        llm_configured: bool = True,
        **options,
    ) -> RAGPipeline:
        retriever = HybridRetriever(store, embedder, reranker=None, use_hybrid=True, candidates_k=10)
        llm = LLMClient("test-model", client=chat_client if llm_configured else None)
        settings = {"chunk_max_tokens": 12, "chunk_overlap_tokens": 4, "top_k": 3} | options
        return RAGPipeline(FakeParser(documents), embedder, store, retriever, llm, **settings)

    return make
