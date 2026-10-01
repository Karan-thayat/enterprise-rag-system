"""Retrieval benchmark on QASPER, plus export of retrieved passages for answer scoring.

A retrieved chunk is relevant when its paragraph was marked as evidence by any
annotator. Only questions with at least one evidence paragraph are scored.

Usage (CPU works for a few papers; the full splits are run on a GPU, see kaggle/):
    python -m eval.run_qasper --split dev --limit-papers 20
"""

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.embeddings import Embedder
from app.retrieval import BM25Index, CrossEncoderReranker
from app.vector_db import RetrievedChunk
from eval.qasper import (
    IndexedPaper,
    Question,
    first_relevant_rank,
    index_papers,
    load_split,
    make_retriever,
    retrieval_metrics,
)

DEPTH = 10  # chunks ranked per question
CANDIDATES = 20  # per first-stage retriever, as in the application
CONTEXT_CHUNKS = 4  # chunks the application passes to the LLM

Ranker = Callable[[IndexedPaper, Question], list[RetrievedChunk]]


def dense_ranker(embedder: Embedder) -> Ranker:
    return lambda indexed, question: indexed.store.query(embedder.embed_query(question.text), DEPTH)


def bm25_ranker() -> Ranker:
    cache: dict[str, BM25Index] = {}

    def rank(indexed: IndexedPaper, question: Question) -> list[RetrievedChunk]:
        index = cache.setdefault(indexed.paper.id, BM25Index(indexed.store.all_chunks()))
        return index.search(question.text, DEPTH)

    return rank


def hybrid_ranker(embedder: Embedder, reranker: CrossEncoderReranker | None = None) -> Ranker:
    """The application's HybridRetriever: dense + BM25 top-20 each, RRF, optional rerank."""

    def rank(indexed: IndexedPaper, question: Question) -> list[RetrievedChunk]:
        retriever = make_retriever(indexed, embedder, reranker, hybrid=True, candidates_k=CANDIDATES)
        return retriever.retrieve(question.text, top_k=DEPTH)

    return rank


def candidates_ranker(embedder: Embedder) -> Ranker:
    """The fused first-stage list: every chunk the reranker gets to reorder."""

    def rank(indexed: IndexedPaper, question: Question) -> list[RetrievedChunk]:
        retriever = make_retriever(indexed, embedder, None, hybrid=True, candidates_k=CANDIDATES)
        return retriever.retrieve(question.text, top_k=CANDIDATES)

    return rank


def evaluate(indexed_papers: list[IndexedPaper], ranker: Ranker, ks: tuple[int, ...] = (1, 4, 10)) -> dict:
    """Returns aggregate metrics plus per-question ranks for significance tests."""
    ranks: dict[str, int | None] = {}
    for indexed in indexed_papers:
        for question in indexed.paper.questions:
            if question.evidence:
                ranks[question.id] = first_relevant_rank(ranker(indexed, question), question.evidence)
    values = list(ranks.values())
    return {"questions": len(values), **retrieval_metrics(values, ks), "ranks": ranks}


def paired_bootstrap(
    baseline: dict[str, int | None],
    candidate: dict[str, int | None],
    metric: Callable[[int | None], float],
    resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean difference (candidate - baseline) of a per-question metric with a 95% bootstrap CI."""
    ids = sorted(baseline)
    diffs = np.array([metric(candidate[i]) - metric(baseline[i]) for i in ids])
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(resamples, len(diffs)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"difference": float(diffs.mean()), "ci_low": float(low), "ci_high": float(high)}


def reciprocal_rank(rank: int | None) -> float:
    return 1 / rank if rank is not None and rank <= DEPTH else 0.0


def hit_at_1(rank: int | None) -> float:
    return float(rank == 1)


def export_passages(
    indexed_papers: list[IndexedPaper], ranker: Ranker, sample_size: int, seed: int = 0
) -> list[dict]:
    """Top chunks for a random sample of questions (answerable or not), for answer scoring."""
    everything = [(indexed, question) for indexed in indexed_papers for question in indexed.paper.questions]
    sample = random.Random(seed).sample(everything, min(sample_size, len(everything)))
    exported = []
    for indexed, question in sample:
        chunks = ranker(indexed, question)[:CONTEXT_CHUNKS]
        exported.append(
            {
                "question_id": question.id,
                "paper_id": indexed.paper.id,
                "question": question.text,
                "references": list(question.references),
                "passages": [
                    {
                        "chunk_id": chunk.id,
                        "paragraph": chunk.page,
                        "text": chunk.text,
                        # The exact string QASPER's evaluator expects as predicted evidence.
                        "evidence_string": indexed.paper.paragraphs[chunk.page],
                    }
                    for chunk in chunks
                ],
            }
        )
    return exported


def summary_row(name: str, result: dict) -> dict:
    return {"name": name, **{k: v for k, v in result.items() if k != "ranks"}}


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    arguments.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    arguments.add_argument("--limit-papers", type=int, default=None)
    arguments.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    arguments.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    arguments.add_argument("--device", default="cpu")
    arguments.add_argument("--output", type=Path, default=None)
    args = arguments.parse_args()

    papers = load_split(args.split)[: args.limit_papers]
    embedder = Embedder(args.embedding_model, device=args.device)
    reranker = CrossEncoderReranker(args.reranker, device=args.device)
    indexed = index_papers(papers, embedder)
    rows = [
        summary_row("Dense only", evaluate(indexed, dense_ranker(embedder))),
        summary_row("BM25 only", evaluate(indexed, bm25_ranker())),
        summary_row("Hybrid (RRF)", evaluate(indexed, hybrid_ranker(embedder))),
        summary_row("Hybrid + rerank", evaluate(indexed, hybrid_ranker(embedder, reranker))),
    ]
    print(json.dumps(rows, indent=2))
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
