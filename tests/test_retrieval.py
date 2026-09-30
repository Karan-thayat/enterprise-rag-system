import pytest

from app.chunking import Chunk
from app.retrieval import BM25Index, HybridRetriever, reciprocal_rank_fusion
from app.vector_db import RetrievedChunk


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(id=chunk_id, text=text, doc_id="doc", filename="doc.pdf", page=1)


def test_reciprocal_rank_fusion_scores_and_order():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60)
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert [item for item, _ in fused] == ["a", "c", "b"]


def test_bm25_returns_only_chunks_that_share_query_terms():
    index = BM25Index(
        [
            make_chunk("0", "The cat sat on the mat."),
            make_chunk("1", "Quarterly revenue grew by 12 percent."),
            make_chunk("2", "Dogs are loyal animals."),
            make_chunk("3", "The weather is sunny today."),
        ]
    )
    hits = index.search("What was the revenue?", k=3)
    assert [hit.id for hit in hits] == ["1"]
    assert hits[0].score > 0


def test_bm25_handles_empty_index_and_empty_query():
    assert BM25Index([]).search("anything", k=3) == []
    assert BM25Index([make_chunk("0", "text")]).search("?!", k=3) == []


def add_facts(store):
    store.add_document(
        "facts",
        "facts.pdf",
        1,
        [
            Chunk("The Eiffel Tower is in Paris.", 1, 0),
            Chunk("Photosynthesis happens in chloroplasts.", 1, 1),
            Chunk("Python was created by Guido van Rossum.", 1, 2),
            Chunk("The Nile is a river in Africa.", 1, 3),
        ],
    )


def test_hybrid_retriever_finds_the_relevant_chunk(store, embedder):
    add_facts(store)
    retriever = HybridRetriever(store, embedder, use_hybrid=True, candidates_k=4)
    results = retriever.retrieve("Who created Python?", top_k=2)
    assert results[0].text == "Python was created by Guido van Rossum."
    assert results[0].filename == "facts.pdf"


class ReversingReranker:
    def rerank(self, query, chunks, top_k):
        return list(reversed(chunks))[:top_k]


def test_reranker_decides_the_final_order(store, embedder):
    add_facts(store)
    plain = HybridRetriever(store, embedder, candidates_k=4).retrieve("Who created Python?", top_k=4)
    reranked = HybridRetriever(store, embedder, reranker=ReversingReranker(), candidates_k=4).retrieve(
        "Who created Python?", top_k=2
    )
    assert [chunk.id for chunk in reranked] == [chunk.id for chunk in reversed(plain)][:2]


def test_refresh_drops_deleted_chunks_from_keyword_search(store, embedder):
    add_facts(store)
    retriever = HybridRetriever(store, embedder, candidates_k=4)
    store.delete_document("facts")
    retriever.refresh()
    assert retriever.retrieve("Who created Python?", top_k=3) == []
