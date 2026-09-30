import re
from itertools import pairwise

import pytest

from app.chunking import chunk_pages
from app.document_parser import Page
from tests.conftest import count_words

SENTENCES = [f"Sentence number {i} has exactly seven words." for i in range(20)]
PAGE_TEXT = " ".join(SENTENCES)
SENTENCE_RE = re.compile(r"Sentence number \d+ has exactly seven words\.")


@pytest.fixture
def chunks():
    return chunk_pages([Page(1, PAGE_TEXT)], max_tokens=30, overlap_tokens=10, count_tokens=count_words)


def test_chunks_respect_the_token_limit(chunks):
    assert len(chunks) > 1
    assert all(count_words(chunk.text) <= 30 for chunk in chunks)


def test_chunks_are_whole_sentences_and_exact_substrings(chunks):
    for chunk in chunks:
        assert chunk.text in PAGE_TEXT
        assert SENTENCE_RE.sub("", chunk.text).strip() == ""


def test_every_sentence_is_covered(chunks):
    for sentence in SENTENCES:
        assert any(sentence in chunk.text for chunk in chunks)


def test_consecutive_chunks_share_the_overlap_sentence(chunks):
    for previous, following in pairwise(chunks):
        last_sentence = SENTENCE_RE.findall(previous.text)[-1]
        assert following.text.startswith(last_sentence)


def test_no_chunk_is_contained_in_its_predecessor(chunks):
    # The original 1,000/200-character window could end with a chunk lying wholly inside the previous one.
    for previous, following in pairwise(chunks):
        assert following.text not in previous.text


def test_page_numbers_and_indices_skip_blank_pages():
    pages = [Page(1, "Alpha beta gamma."), Page(2, "   \n "), Page(3, "Delta epsilon zeta.")]
    chunks = chunk_pages(pages, max_tokens=10, overlap_tokens=2, count_tokens=count_words)
    assert [(chunk.page, chunk.index, chunk.text) for chunk in chunks] == [
        (1, 0, "Alpha beta gamma."),
        (3, 1, "Delta epsilon zeta."),
    ]


def test_oversized_sentence_is_split_at_word_boundaries():
    sentence = " ".join(["word"] * 95) + "."
    chunks = chunk_pages([Page(1, sentence)], max_tokens=30, overlap_tokens=5, count_tokens=count_words)
    assert [count_words(chunk.text) for chunk in chunks] == [30, 30, 30, 5]
    assert " ".join(chunk.text for chunk in chunks) == sentence


def test_layout_whitespace_inside_a_chunk_is_preserved():
    table = "Model      Perplexity\nBaseline   11.54\nMELODI     10.44"
    chunks = chunk_pages([Page(1, table)], max_tokens=50, overlap_tokens=5, count_tokens=count_words)
    assert [chunk.text for chunk in chunks] == [table]


@pytest.mark.parametrize("max_tokens, overlap_tokens", [(0, 0), (10, 10), (10, -1)])
def test_invalid_sizes_are_rejected(max_tokens, overlap_tokens):
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], max_tokens, overlap_tokens, count_tokens=count_words)
