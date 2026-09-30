import pytest

from app.llm_generator import REWRITE_SYSTEM_PROMPT, LLMNotConfiguredError
from app.pipeline import NO_DOCUMENTS_ANSWER, EmptyDocumentError
from tests.conftest import BIOLOGY_PDF, BLANK_PDF, EIFFEL_PDF, PYTHON_PDF


def test_ingest_indexes_pages_and_deduplicates_by_content(make_pipeline):
    pipeline = make_pipeline()
    first = pipeline.ingest_pdf(EIFFEL_PDF, "eiffel.pdf")
    again = pipeline.ingest_pdf(EIFFEL_PDF, "renamed.pdf")

    assert not first.already_indexed
    assert (first.document.filename, first.document.pages) == ("eiffel.pdf", 2)
    assert first.document.chunks == 4
    assert again.already_indexed
    assert again.document == first.document
    assert pipeline._parser.calls == 1


def test_documents_with_no_text_are_rejected(make_pipeline):
    pipeline = make_pipeline()
    with pytest.raises(EmptyDocumentError):
        pipeline.ingest_pdf(BLANK_PDF, "blank.pdf")
    assert pipeline.count_chunks() == 0


def test_answer_without_documents_skips_the_llm(make_pipeline, chat_client):
    result = make_pipeline().answer("Anything?", [])
    assert result.answer == NO_DOCUMENTS_ANSWER
    assert result.sources == []
    assert chat_client.calls == []


def test_answer_requires_a_configured_llm(make_pipeline):
    pipeline = make_pipeline(llm_configured=False)
    pipeline.ingest_pdf(EIFFEL_PDF, "eiffel.pdf")
    with pytest.raises(LLMNotConfiguredError):
        pipeline.answer("How tall is the tower?", [])


def test_first_question_is_answered_from_retrieved_sources(make_pipeline, chat_client):
    pipeline = make_pipeline()
    for pdf, name in [(EIFFEL_PDF, "eiffel.pdf"), (BIOLOGY_PDF, "biology.pdf"), (PYTHON_PDF, "python.pdf")]:
        pipeline.ingest_pdf(pdf, name)

    result = pipeline.answer("Who created Python?", [])

    assert result.standalone_question == "Who created Python?"
    assert result.answer == "Stub answer [1]."
    assert result.sources[0].text == "Python was created by Guido van Rossum."
    assert (result.sources[0].filename, result.sources[0].page) == ("python.pdf", 1)
    assert len(chat_client.calls) == 1  # no rewrite without history
    assert set(result.timings_ms) == {"retrieval", "generation", "total"}


def test_follow_up_is_rewritten_before_retrieval(make_pipeline, chat_client):
    pipeline = make_pipeline()
    pipeline.ingest_pdf(EIFFEL_PDF, "eiffel.pdf")
    pipeline.ingest_pdf(PYTHON_PDF, "python.pdf")
    chat_client.replies = ["How tall is the Eiffel Tower?", "It is 330 metres tall [1]."]
    history = [
        {"role": "user", "content": "Where is the Eiffel Tower?"},
        {"role": "assistant", "content": "In Paris [1]."},
    ]

    result = pipeline.answer("How tall is it?", history)

    rewrite_call, answer_call = chat_client.calls
    assert rewrite_call["messages"][0]["content"] == REWRITE_SYSTEM_PROMPT
    assert result.standalone_question == "How tall is the Eiffel Tower?"
    assert result.sources[0].text == "The tower is 330 metres tall."
    assert answer_call["messages"][1:3] == history
    assert answer_call["messages"][-1]["content"].endswith("Question: How tall is it?")
    assert "rewrite" in result.timings_ms


def test_history_is_truncated_to_the_most_recent_messages(make_pipeline, chat_client):
    pipeline = make_pipeline(max_history_messages=2, rewrite_follow_ups=False)
    pipeline.ingest_pdf(EIFFEL_PDF, "eiffel.pdf")
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(6)]

    pipeline.answer("How tall is the tower?", history)

    messages = chat_client.calls[0]["messages"]
    assert messages[1:-1] == history[-2:]


def test_delete_document_removes_it_from_search(make_pipeline):
    pipeline = make_pipeline()
    result = pipeline.ingest_pdf(PYTHON_PDF, "python.pdf")
    pipeline.ingest_pdf(EIFFEL_PDF, "eiffel.pdf")

    assert pipeline.delete_document(result.document.doc_id)
    assert not pipeline.delete_document(result.document.doc_id)
    assert [document.filename for document in pipeline.list_documents()] == ["eiffel.pdf"]
    assert all(chunk.filename == "eiffel.pdf" for chunk in pipeline.search("Who created Python?", top_k=3))
