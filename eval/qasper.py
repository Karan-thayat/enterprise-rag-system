"""QASPER: a public benchmark of questions about NLP research papers (Dasigi et al., NAACL 2021).

Each question is asked about one paper, and annotators marked the paragraphs that
answer it. This module loads the official JSON release and runs the application's
own retrieval stack (chunker, embedder, BM25, RRF, reranker) over each paper.
Only the vector store differs: each paper gets a small in-memory index instead
of a Chroma collection, which is equivalent for a few hundred chunks.

Download the data (CC BY 4.0) into eval/.cache/qasper/:
    https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz
    https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from app.chunking import Chunk, chunk_pages
from app.document_parser import Page
from app.embeddings import Embedder
from app.retrieval import CrossEncoderReranker, HybridRetriever
from app.vector_db import RetrievedChunk

DATA_DIR = Path(__file__).parent / ".cache" / "qasper"
SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}
FLOAT_PREFIX = "FLOAT SELECTED: "


def normalize_space(text: str) -> str:
    return " ".join(text.split())


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    evidence: frozenset[int]  # ids of paragraphs any annotator marked as evidence
    references: tuple[dict, ...]  # the official evaluator's per-annotator {answer, evidence, type}


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    paragraphs: tuple[str, ...]  # body paragraphs, then "FLOAT SELECTED: <caption>" entries
    questions: tuple[Question, ...]

    def paragraph_text(self, paragraph_id: int) -> str:
        """Text to index: figure and table captions without the evaluator's prefix."""
        return self.paragraphs[paragraph_id].removeprefix(FLOAT_PREFIX)


def references_for(answers: list[dict]) -> tuple[dict, ...]:
    """Per-annotator references, built exactly as QASPER's official evaluator builds them."""
    references = []
    for annotation in answers:
        answer = annotation["answer"]
        if answer["unanswerable"]:
            references.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
            continue
        if answer["extractive_spans"]:
            text, kind = ", ".join(answer["extractive_spans"]), "extractive"
        elif answer["free_form_answer"]:
            text, kind = answer["free_form_answer"], "abstractive"
        elif answer["yes_no"] is not None:
            text, kind = ("Yes" if answer["yes_no"] else "No"), "boolean"
        else:
            raise ValueError(f"annotation {answer.get('annotation_id')} has no answer")
        references.append({"answer": text, "evidence": answer["evidence"], "type": kind})
    return tuple(references)


def parse_papers(data: dict) -> list[Paper]:
    papers = []
    for paper_id, raw in data.items():
        paragraphs = [text for section in raw["full_text"] for text in section["paragraphs"] if text.strip()]
        paragraphs += [FLOAT_PREFIX + figure["caption"] for figure in raw["figures_and_tables"]]
        ids = {}
        for index, text in enumerate(paragraphs):
            ids.setdefault(normalize_space(text), index)
        questions = []
        for qa in raw["qas"]:
            evidence = {
                ids[normalize_space(text)]
                for annotation in qa["answers"]
                for text in annotation["answer"]["evidence"]
                # Evidence strings that are section headings ("A ::: B") match no paragraph.
                if normalize_space(text) in ids
            }
            questions.append(
                Question(
                    qa["question_id"],
                    qa["question"].strip(),
                    frozenset(evidence),
                    references_for(qa["answers"]),
                )
            )
        papers.append(Paper(paper_id, raw["title"], tuple(paragraphs), tuple(questions)))
    return papers


def load_split(split: str, data_dir: Path = DATA_DIR) -> list[Paper]:
    path = data_dir / SPLIT_FILES[split]
    if not path.exists():
        raise SystemExit(f"{path} not found; see the download links at the top of eval/qasper.py")
    return parse_papers(json.loads(path.read_text()))


def chunk_paper(paper: Paper, max_tokens: int, overlap_tokens: int, count_tokens: Callable) -> list[Chunk]:
    """Chunks each paragraph separately; a chunk's ``page`` is its paragraph id."""
    pages = [Page(number=index, text=paper.paragraph_text(index)) for index in range(len(paper.paragraphs))]
    return chunk_pages(pages, max_tokens, overlap_tokens, count_tokens)


class InMemoryStore:
    """The slice of VectorStore that HybridRetriever uses, backed by a numpy matrix."""

    def __init__(self, paper_id: str, chunks: list[Chunk], vectors: np.ndarray):
        self._chunks = [
            RetrievedChunk(
                id=str(chunk.index), text=chunk.text, doc_id=paper_id, filename=paper_id, page=chunk.page
            )
            for chunk in chunks
        ]
        self._vectors = vectors

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[RetrievedChunk]:
        return list(self._chunks)

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        if not self._chunks:
            return []
        scores = self._vectors @ np.asarray(embedding)  # cosine: both sides are unit length
        top = np.argsort(-scores, kind="stable")[:k]
        return [replace(self._chunks[i], score=float(scores[i])) for i in top]


@dataclass(frozen=True)
class IndexedPaper:
    paper: Paper
    chunks: list[Chunk]
    store: InMemoryStore


def index_papers(
    papers: Iterable[Paper], embedder: Embedder, max_tokens: int = 200, overlap_tokens: int = 40
) -> list[IndexedPaper]:
    """Chunks and embeds every paper in one batch (much faster than paper by paper)."""
    papers = list(papers)
    chunked = [chunk_paper(paper, max_tokens, overlap_tokens, embedder.count_tokens) for paper in papers]
    vectors = np.asarray(embedder.embed_documents([chunk.text for chunks in chunked for chunk in chunks]))
    indexed, offset = [], 0
    for paper, chunks in zip(papers, chunked, strict=True):
        paper_vectors = vectors[offset : offset + len(chunks)]
        offset += len(chunks)
        indexed.append(IndexedPaper(paper, chunks, InMemoryStore(paper.id, chunks, paper_vectors)))
    return indexed


def make_retriever(
    indexed: IndexedPaper,
    embedder: Embedder,
    reranker: CrossEncoderReranker | None,
    hybrid: bool,
    candidates_k: int = 20,
) -> HybridRetriever:
    return HybridRetriever(indexed.store, embedder, reranker, use_hybrid=hybrid, candidates_k=candidates_k)


def first_relevant_rank(results: list[RetrievedChunk], evidence: frozenset[int]) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if chunk.page in evidence:
            return rank
    return None


def retrieval_metrics(ranks: list[int | None], ks: tuple[int, ...] = (1, 4, 10)) -> dict[str, float]:
    metrics = {f"Hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in ks}
    metrics["MRR@10"] = sum(1 / r for r in ranks if r is not None and r <= 10) / len(ranks)
    return metrics
