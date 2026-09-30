import hashlib
import logging
import threading
import time
from dataclasses import dataclass

from app.chunking import chunk_pages
from app.config import Settings
from app.document_parser import DocumentParser
from app.embeddings import Embedder
from app.llm_generator import LLMClient, LLMNotConfiguredError
from app.retrieval import CrossEncoderReranker, HybridRetriever
from app.vector_db import DocumentSummary, RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

NO_DOCUMENTS_ANSWER = "No documents have been indexed yet. Upload a PDF first."


class EmptyDocumentError(ValueError):
    """Raised when a PDF contains no extractable text."""


@dataclass(frozen=True)
class IngestResult:
    document: DocumentSummary
    already_indexed: bool


@dataclass(frozen=True)
class Answer:
    question: str
    standalone_question: str
    answer: str
    sources: list[RetrievedChunk]
    timings_ms: dict[str, float]


class RAGPipeline:
    """Ingestion (parse -> chunk -> embed -> index) and question answering
    (rewrite -> hybrid retrieval -> rerank -> grounded generation)."""

    def __init__(
        self,
        parser: DocumentParser,
        embedder: Embedder,
        store: VectorStore,
        retriever: HybridRetriever,
        llm: LLMClient,
        *,
        chunk_max_tokens: int,
        chunk_overlap_tokens: int,
        top_k: int,
        rewrite_follow_ups: bool = True,
        max_history_messages: int = 6,
    ):
        self._parser = parser
        self._embedder = embedder
        self._store = store
        self._retriever = retriever
        self._llm = llm
        self.chunk_max_tokens = min(chunk_max_tokens, embedder.max_tokens)
        self.chunk_overlap_tokens = min(chunk_overlap_tokens, self.chunk_max_tokens - 1)
        self.top_k = top_k
        self.rewrite_follow_ups = rewrite_follow_ups
        self.max_history_messages = max_history_messages
        # Serialises writes so the BM25 index is always rebuilt from a consistent store.
        self._write_lock = threading.Lock()

    @property
    def llm_configured(self) -> bool:
        return self._llm.configured

    def ingest_pdf(self, data: bytes, filename: str) -> IngestResult:
        """Indexes a PDF. Documents are identified by a hash of their bytes, so
        re-uploading the same file is a no-op."""
        doc_id = hashlib.sha256(data).hexdigest()[:16]
        with self._write_lock:
            existing = self._store.get_document(doc_id)
            if existing is not None:
                return IngestResult(document=existing, already_indexed=True)

            started = time.perf_counter()
            pages = self._parser.parse(data)
            chunks = chunk_pages(
                pages,
                max_tokens=self.chunk_max_tokens,
                overlap_tokens=self.chunk_overlap_tokens,
                count_tokens=self._embedder.count_tokens,
            )
            if not chunks:
                raise EmptyDocumentError("No extractable text was found in this PDF.")
            self._store.add_document(doc_id, filename, len(pages), chunks)
            self._retriever.refresh()

        logger.info(
            "Indexed %s (%d pages, %d chunks) in %.1fs",
            filename,
            len(pages),
            len(chunks),
            time.perf_counter() - started,
        )
        return IngestResult(
            document=DocumentSummary(doc_id, filename, len(pages), len(chunks)),
            already_indexed=False,
        )

    def delete_document(self, doc_id: str) -> bool:
        with self._write_lock:
            deleted = self._store.delete_document(doc_id)
            if deleted:
                self._retriever.refresh()
        return deleted > 0

    def list_documents(self) -> list[DocumentSummary]:
        return self._store.list_documents()

    def count_chunks(self) -> int:
        return self._store.count()

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return self._retriever.retrieve(query, top_k)

    def answer(self, question: str, history: list[dict[str, str]]) -> Answer:
        started = time.perf_counter()
        if self._store.count() == 0:
            return Answer(question, question, NO_DOCUMENTS_ANSWER, [], {"total": _ms(started)})
        if not self._llm.configured:
            raise LLMNotConfiguredError("Set GROQ_API_KEY on the server to enable answers.")

        history = history[-self.max_history_messages :] if self.max_history_messages > 0 else []
        timings: dict[str, float] = {}

        # Follow-ups like "what about its limitations?" retrieve poorly on their own,
        # so they are rewritten into standalone queries first.
        standalone = question
        if history and self.rewrite_follow_ups:
            step = time.perf_counter()
            standalone = self._llm.rewrite_question(question, history)
            timings["rewrite"] = _ms(step)

        step = time.perf_counter()
        sources = self._retriever.retrieve(standalone, self.top_k)
        timings["retrieval"] = _ms(step)

        step = time.perf_counter()
        answer = self._llm.generate_answer(question, sources, history)
        timings["generation"] = _ms(step)
        timings["total"] = _ms(started)

        return Answer(question, standalone, answer, sources, timings)


def build_pipeline(settings: Settings) -> RAGPipeline:
    """Loads the models and opens the vector store described by ``settings``."""
    embedder = Embedder(settings.embedding_model)
    store = VectorStore(settings.chroma_path, settings.collection_name, embedder)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
    retriever = HybridRetriever(
        store,
        embedder,
        reranker,
        use_hybrid=settings.use_hybrid,
        candidates_k=settings.candidates_k,
    )
    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    llm = LLMClient(
        settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
        api_key=api_key,
    )
    parser = DocumentParser(
        ocr_enabled=settings.ocr_enabled,
        timeout_s=settings.parse_timeout_s,
        cli_path=settings.liteparse_cli,
    )
    return RAGPipeline(
        parser,
        embedder,
        store,
        retriever,
        llm,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
        top_k=settings.top_k,
        rewrite_follow_ups=settings.rewrite_follow_ups,
        max_history_messages=settings.max_history_messages,
    )


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)
