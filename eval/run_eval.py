"""Retrieval evaluation on a hand-labelled question set.

Compares the original pipeline (1,000-character windows, dense search, top-3)
with the upgraded one, then sweeps the chunk size. A retrieved chunk counts as
relevant when it contains one of the question's evidence phrases (compared
case- and whitespace-insensitively). Every phrase is first checked against the
parsed PDF, so a labelling mistake fails loudly instead of skewing the numbers.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --pdf my.pdf --golden my_questions.jsonl --output my_results.md
"""

import argparse
import hashlib
import json
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from app.chunking import Chunk, chunk_pages
from app.config import get_settings
from app.document_parser import DocumentParser, Page
from app.embeddings import Embedder
from app.retrieval import CrossEncoderReranker, HybridRetriever
from app.vector_db import RetrievedChunk, VectorStore

EVAL_DIR = Path(__file__).parent
DEFAULT_PDF = EVAL_DIR / "data" / "melodi_iclr2025.pdf"
DEFAULT_GOLDEN = EVAL_DIR / "golden_set.jsonl"
DEFAULT_OUTPUT = EVAL_DIR / "results.md"
CACHE_DIR = EVAL_DIR / ".cache"
HIT_KS = (1, 3, 5)
DEPTH = 10  # ranks considered for MRR


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    evidence: list[tuple[int, str]]  # (page, normalised phrase)


@dataclass(frozen=True)
class Config:
    name: str
    chunker: Callable[[list[Page]], list[Chunk]]
    hybrid: bool
    rerank: bool


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text().splitlines():
        if line.strip():
            item = json.loads(line)
            evidence = [(e["page"], normalize(e["text"])) for e in item["evidence"]]
            questions.append(Question(item["id"], item["question"], evidence))
    return questions


def validate_evidence(questions: list[Question], pages: list[Page]) -> None:
    page_text = {page.number: normalize(page.text) for page in pages}
    missing = [
        f"{q.id}: {phrase!r} not found on page {page}"
        for q in questions
        for page, phrase in q.evidence
        if phrase not in page_text.get(page, "")
    ]
    if missing:
        raise SystemExit("Golden set does not match the PDF:\n" + "\n".join(missing))


def parse_pdf(path: Path) -> list[Page]:
    """Parses the PDF with the production parser, caching the result on disk."""
    settings = get_settings()
    data = path.read_bytes()
    cache = CACHE_DIR / f"{hashlib.sha256(data).hexdigest()[:16]}-ocr{int(settings.ocr_enabled)}.json"
    if cache.exists():
        return [Page(**page) for page in json.loads(cache.read_text())]
    parser = DocumentParser(
        ocr_enabled=settings.ocr_enabled, timeout_s=settings.parse_timeout_s, cli_path=settings.liteparse_cli
    )
    pages = parser.parse(data)
    CACHE_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps([asdict(page) for page in pages]))
    return pages


def original_chunker(pages: list[Page]) -> list[Chunk]:
    """The original chunker: 1,000-character windows, 200-character overlap, no page tracking."""
    text = "\n\n".join(page.text for page in pages)  # identical to LiteParse's `result.text`
    starts = range(0, len(text), 1000 - 200)
    return [Chunk(text=text[start : start + 1000], page=0, index=i) for i, start in enumerate(starts)]


def sentence_chunker(embedder: Embedder, max_tokens: int, overlap_tokens: int):
    def chunker(pages: list[Page]) -> list[Chunk]:
        return chunk_pages(pages, max_tokens, overlap_tokens, embedder.count_tokens)

    return chunker


def first_relevant_rank(results: list[RetrievedChunk], question: Question) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        text = normalize(chunk.text)
        if any(phrase in text for _, phrase in question.evidence):
            return rank
    return None


def evaluate(
    config: Config,
    pages: list[Page],
    questions: list[Question],
    embedder: Embedder,
    reranker: CrossEncoderReranker,
) -> dict:
    chunks = config.chunker(pages)
    token_counts = [embedder.count_tokens(chunk.text) for chunk in chunks]
    chunk_texts = [normalize(chunk.text) for chunk in chunks]
    # Upper bound: questions whose evidence sits wholly inside at least one chunk.
    coverage = sum(
        any(phrase in text for _, phrase in q.evidence for text in chunk_texts) for q in questions
    ) / len(questions)

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(tmp, "eval", embedder)
        store.add_document("eval", "eval.pdf", len(pages), chunks)
        retriever = HybridRetriever(
            store,
            embedder,
            reranker if config.rerank else None,
            use_hybrid=config.hybrid,
            candidates_k=20,
        )
        ranks, latencies = [], []
        for question in questions:
            started = time.perf_counter()
            results = retriever.retrieve(question.question, top_k=DEPTH)
            latencies.append((time.perf_counter() - started) * 1000)
            ranks.append(first_relevant_rank(results, question))

    metrics = {f"Hit@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in HIT_KS}
    metrics["MRR@10"] = sum(1 / r for r in ranks if r is not None) / len(ranks)
    return {
        "name": config.name,
        "chunks": len(chunks),
        "mean_tokens": statistics.mean(token_counts),
        "truncated": sum(n > embedder.max_tokens for n in token_counts) / len(chunks),
        "coverage": coverage,
        "latency_ms": statistics.median(latencies),
        "misses": [q.id for q, r in zip(questions, ranks, strict=True) if r is None or r > HIT_KS[-1]],
        **metrics,
    }


def table(rows: list[dict]) -> str:
    header = (
        "| Configuration | Chunks | Mean tokens | Truncated | Hit@1 | Hit@3 | Hit@5 | MRR@10 "
        "| Median latency |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [
        f"| {r['name']} | {r['chunks']} | {r['mean_tokens']:.0f} | {r['truncated']:.0%} "
        f"| {r['Hit@1']:.1%} | {r['Hit@3']:.1%} | {r['Hit@5']:.1%} | {r['MRR@10']:.3f} "
        f"| {r['latency_ms']:.0f} ms |"
        for r in rows
    ]
    return "\n".join([header, *lines])


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    arguments.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    arguments.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    arguments.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments.add_argument("--title", default="Retrieval evaluation")
    args = arguments.parse_args()

    settings = get_settings()
    pages = parse_pdf(args.pdf)
    questions = load_questions(args.golden)
    validate_evidence(questions, pages)
    embedder = Embedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model)
    size, overlap = settings.chunk_max_tokens, settings.chunk_overlap_tokens
    default_chunker = sentence_chunker(embedder, size, overlap)

    def run(name: str, chunker, hybrid: bool, rerank: bool) -> dict:
        return evaluate(Config(name, chunker, hybrid, rerank), pages, questions, embedder, reranker)

    retrieval_rows = [
        run("Original: 1,000-char windows, dense only", original_chunker, False, False),
        run(f"Sentence chunks ({size} tok), dense only", default_chunker, False, False),
        run("+ BM25 hybrid (RRF)", default_chunker, True, False),
        run("+ cross-encoder rerank (default)", default_chunker, True, True),
    ]
    chunking_rows = [run("1,000-char windows (original)", original_chunker, True, True)]
    for n, o in [(128, 24), (200, 40), (254, 48)]:
        name = f"Sentence chunks, {n} tok / {o} overlap"
        if (n, o) == (size, overlap):  # already measured above; reuse it so both tables agree
            chunking_rows.append(retrieval_rows[-1] | {"name": f"{name} (default)"})
        else:
            chunking_rows.append(run(name, sentence_chunker(embedder, n, o), True, True))

    split = [
        f"{r['name']} ({r['coverage']:.0%})" for r in retrieval_rows + chunking_rows if r["coverage"] < 1
    ]
    if split:
        coverage_note = (
            "Some evidence phrases are split across chunk boundaries in: " + "; ".join(split) + "."
        )
    else:
        coverage_note = (
            "In every configuration each question's evidence lies wholly inside at least one chunk, "
            "so every question is answerable in principle."
        )
    misses = ", ".join(retrieval_rows[-1]["misses"]) or "none"

    lines = [
        f"# {args.title}",
        "",
        f"- **Corpus:** `{args.pdf.name}` ({len(pages)} pages), parsed with LiteParse "
        f"(OCR {'on' if settings.ocr_enabled else 'off'})",
        f"- **Questions:** {len(questions)} hand-written questions in `{args.golden.name}`, each labelled "
        "with verbatim evidence phrases that are checked against the parsed text",
        "- **Relevance:** a retrieved chunk is relevant if it contains an evidence phrase. Hit@k is the "
        "share of questions with a relevant chunk in the top k; MRR@10 is the mean reciprocal rank of "
        "the first relevant chunk",
        f"- **Models:** `{settings.embedding_model}` embeddings, `{settings.reranker_model}` reranker, "
        "20 candidates per retriever before fusion",
        f"- **Run:** {date.today().isoformat()}, CPU only ({cpu_name()}). Latency is the median per "
        "question and excludes the LLM call",
        f"- **Truncated:** share of chunks longer than the embedder's {embedder.max_tokens}-token input; "
        "their tails are silently ignored when embedded",
        "",
        "## Retrieval ablation",
        "",
        "The original system passed its top 3 chunks to the LLM; "
        f"the upgraded one passes its top {settings.top_k}.",
        "",
        table(retrieval_rows),
        "",
        "## Chunking ablation (hybrid retrieval + reranking)",
        "",
        table(chunking_rows),
        "",
        coverage_note,
        "",
        f"Questions the default configuration does not retrieve in the top 5: {misses}.",
        "",
        "Reproduce with `python -m eval.run_eval`.",
    ]
    report = "\n".join(lines) + "\n"
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
