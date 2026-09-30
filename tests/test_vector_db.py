from app.chunking import Chunk
from app.vector_db import DocumentSummary


def test_documents_do_not_overwrite_each_other(store):
    # Regression test: IDs used to be chunk_0..chunk_n for every upload, so a second
    # document silently replaced the first chunks of the previous one.
    store.add_document("doc-a", "a.pdf", 1, [Chunk(f"alpha {i}", 1, i) for i in range(3)])
    store.add_document("doc-b", "b.pdf", 1, [Chunk("beta 0", 1, 0)])

    assert store.count() == 4
    assert {chunk.text for chunk in store.all_chunks()} == {"alpha 0", "alpha 1", "alpha 2", "beta 0"}


def test_adding_the_same_document_twice_is_idempotent(store):
    chunks = [Chunk("same text", 1, 0), Chunk("more text", 2, 1)]
    store.add_document("doc-a", "a.pdf", 2, chunks)
    store.add_document("doc-a", "a.pdf", 2, chunks)
    assert store.count() == 2


def test_list_get_and_delete_documents(store):
    store.add_document("doc-b", "b.pdf", 5, [Chunk("beta", 4, 0)])
    store.add_document("doc-a", "a.pdf", 2, [Chunk("alpha", 1, 0), Chunk("alpha two", 2, 1)])

    assert store.list_documents() == [
        DocumentSummary("doc-a", "a.pdf", pages=2, chunks=2),
        DocumentSummary("doc-b", "b.pdf", pages=5, chunks=1),
    ]
    assert store.get_document("doc-b") == DocumentSummary("doc-b", "b.pdf", pages=5, chunks=1)
    assert store.get_document("missing") is None

    assert store.delete_document("doc-a") == 2
    assert store.delete_document("doc-a") == 0
    assert [document.doc_id for document in store.list_documents()] == ["doc-b"]


def test_query_returns_citation_metadata_and_cosine_scores(store, embedder):
    store.add_document(
        "doc-a",
        "a.pdf",
        3,
        [Chunk("The Eiffel Tower is in Paris.", 3, 0), Chunk("Bananas are yellow.", 1, 1)],
    )
    results = store.query(embedder.embed_query("Where is the Eiffel Tower?"), k=5)

    assert len(results) == 2
    top = results[0]
    assert (top.text, top.filename, top.page, top.doc_id) == (
        "The Eiffel Tower is in Paris.",
        "a.pdf",
        3,
        "doc-a",
    )
    assert 0 < top.score <= 1
    assert top.score > results[1].score


def test_query_on_an_empty_store_returns_nothing(store, embedder):
    assert store.query(embedder.embed_query("anything"), k=3) == []
