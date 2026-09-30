import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from app.document_parser import Page

# Sentence ends (., !, ? followed by whitespace) and blank lines delimit the units chunks are built from.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_LINE_BREAK = re.compile(r"\n")
_WORD = re.compile(r"\S+")

# (start, end, n_tokens): a slice of the page text and its token count.
Span = tuple[int, int, int]


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int  # position of the chunk within its document


def chunk_pages(
    pages: Iterable[Page],
    max_tokens: int,
    overlap_tokens: int,
    count_tokens: Callable[[str], int],
) -> list[Chunk]:
    """Splits each page into chunks of at most ``max_tokens`` tokens.

    Chunks are packed from whole sentences (an oversized sentence falls back to
    line, then word, boundaries). Consecutive chunks on a page share up to
    ``overlap_tokens`` tokens of trailing sentences. Every chunk is an exact
    substring of its page's text and never spans two pages, so it can be cited
    with a single page number.

    Token counts are summed per sentence, which is exact for tokenizers that
    split on whitespace first (BERT-style WordPiece, as used by MiniLM).
    """
    if max_tokens <= 0 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("Require max_tokens > 0 and 0 <= overlap_tokens < max_tokens.")

    chunks: list[Chunk] = []
    for page in pages:
        for text in _chunk_text(page.text, max_tokens, overlap_tokens, count_tokens):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks


def _chunk_text(
    text: str, max_tokens: int, overlap_tokens: int, count_tokens: Callable[[str], int]
) -> list[str]:
    units: list[Span] = []
    for start, end in _segments(text, 0, len(text), _SENTENCE_BOUNDARY):
        n_tokens = count_tokens(text[start:end])
        if n_tokens <= max_tokens:
            units.append((start, end, n_tokens))
        else:
            units.extend(_split_oversized(text, start, end, max_tokens, count_tokens))

    chunks: list[str] = []
    window: list[Span] = []
    for unit in units:
        if window and _size(window) + unit[2] > max_tokens:
            chunks.append(text[window[0][0] : window[-1][1]])
            # Carry the trailing sentences over as overlap, then make room for the new unit.
            window = _tail(window, overlap_tokens)
            while window and _size(window) + unit[2] > max_tokens:
                window.pop(0)
        window.append(unit)
    if window:
        chunks.append(text[window[0][0] : window[-1][1]])
    return chunks


def _split_oversized(
    text: str, start: int, end: int, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[Span]:
    """Splits a span longer than ``max_tokens`` at line breaks, then at spaces."""
    pieces: list[Span] = []
    for line_start, line_end in _segments(text, start, end, _LINE_BREAK):
        n_tokens = count_tokens(text[line_start:line_end])
        if n_tokens <= max_tokens:
            pieces.append((line_start, line_end, n_tokens))
        else:
            # A single "word" longer than max_tokens stays whole; the embedder truncates it.
            pieces.extend(
                (match.start(), match.end(), count_tokens(match.group()))
                for match in _WORD.finditer(text, line_start, line_end)
            )

    # Re-join neighbouring pieces greedily while they still fit.
    merged: list[Span] = []
    for piece in pieces:
        if merged and merged[-1][2] + piece[2] <= max_tokens:
            merged[-1] = (merged[-1][0], piece[1], merged[-1][2] + piece[2])
        else:
            merged.append(piece)
    return merged


def _segments(text: str, start: int, end: int, separator: re.Pattern) -> list[tuple[int, int]]:
    """Returns the non-blank spans of text[start:end] between separator matches, trimmed."""
    spans = []
    position = start
    for match in separator.finditer(text, start, end):
        spans.append((position, match.start()))
        position = match.end()
    spans.append((position, end))
    return [_strip(text, s, e) for s, e in spans if text[s:e].strip()]


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _tail(window: list[Span], budget: int) -> list[Span]:
    """Returns the longest suffix of ``window`` whose token total fits in ``budget``."""
    tail: list[Span] = []
    total = 0
    for unit in reversed(window):
        if total + unit[2] > budget:
            break
        tail.insert(0, unit)
        total += unit[2]
    return tail


def _size(window: list[Span]) -> int:
    return sum(unit[2] for unit in window)
