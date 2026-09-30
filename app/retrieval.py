import re
from collections import defaultdict
from dataclasses import replace

import numpy as np
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder
from app.vector_db import RetrievedChunk, VectorStore

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuses ranked ID lists with RRF: score(d) = sum over rankings of 1 / (k + rank(d)).

    RRF only uses ranks, so it can combine retrievers whose scores are on
    different scales (cosine similarity vs. BM25). k=60 follows Cormack et al. (2009).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


class BM25Index:
    """In-memory BM25 keyword index over every stored chunk."""

    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        terms = tokenize(query)
        if self._bm25 is None or not terms:
            return []
        scores = self._bm25.get_scores(terms)
        top = np.argsort(-scores, kind="stable")[:k]
        return [replace(self._chunks[i], score=float(scores[i])) for i in top if scores[i] > 0]


class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs jointly with a cross-encoder, which is slower
    but more accurate than comparing independently computed embeddings."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return []
        scores = np.asarray(self._model.predict([(query, chunk.text) for chunk in chunks]))
        top = np.argsort(-scores, kind="stable")[:top_k]
        return [replace(chunks[i], score=float(scores[i])) for i in top]


class HybridRetriever:
    """Dense + BM25 candidate generation, fused with RRF, then optional reranking."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None = None,
        use_hybrid: bool = True,
        candidates_k: int = 20,
    ):
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self.use_hybrid = use_hybrid
        self.candidates_k = candidates_k
        self._bm25 = BM25Index([])
        self.refresh()

    def refresh(self) -> None:
        """Rebuilds the BM25 index from the vector store; call after adding or deleting documents."""
        if self.use_hybrid:
            self._bm25 = BM25Index(self._store.all_chunks())

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidates = self._store.query(self._embedder.embed_query(query), self.candidates_k)
        if self.use_hybrid:
            keyword_hits = self._bm25.search(query, self.candidates_k)
            by_id = {chunk.id: chunk for chunk in keyword_hits + candidates}
            fused = reciprocal_rank_fusion(
                [[chunk.id for chunk in candidates], [chunk.id for chunk in keyword_hits]]
            )
            candidates = [replace(by_id[chunk_id], score=score) for chunk_id, score in fused]
            candidates = candidates[: self.candidates_k]
        if self._reranker is not None:
            return self._reranker.rerank(query, candidates, top_k)
        return candidates[:top_k]
