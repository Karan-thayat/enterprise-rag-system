"""Hard-negative mining for fine-tuning the retrieval models on QASPER.

For each training question, positives are the chunks of its evidence paragraphs and
hard negatives are the best-ranked non-evidence chunks from hybrid retrieval, i.e.
the passages the application actually confuses with the answer.

Optionally a cross-encoder "teacher" drops negatives it ranks above every marked
evidence chunk: QASPER annotators mark some but not all supporting paragraphs, so a
top-ranked "negative" is sometimes unmarked evidence (the denoising idea from
RocketQA, Qu et al., 2021). Comparing ranks rather than a score threshold keeps the
rule valid whatever activation the cross-encoder applies.
"""

from dataclasses import dataclass

from app.embeddings import Embedder
from app.retrieval import CrossEncoderReranker
from eval.qasper import IndexedPaper, make_retriever


@dataclass(frozen=True)
class Example:
    question: str
    positives: tuple[str, ...]
    negatives: tuple[str, ...]


def mine_examples(
    indexed_papers: list[IndexedPaper],
    embedder: Embedder,
    teacher: CrossEncoderReranker | None = None,
    max_negatives: int = 5,
    candidates_k: int = 20,
) -> list[Example]:
    examples = []
    for indexed in indexed_papers:
        retriever = make_retriever(indexed, embedder, reranker=None, hybrid=True, candidates_k=candidates_k)
        for question in indexed.paper.questions:
            if not question.evidence:
                continue
            positives = tuple(chunk.text for chunk in indexed.chunks if chunk.page in question.evidence)
            candidates = [
                chunk
                for chunk in retriever.retrieve(question.text, top_k=candidates_k)
                if chunk.page not in question.evidence
            ]
            if teacher is not None and candidates:
                evidence = [chunk for chunk in indexed.store.all_chunks() if chunk.page in question.evidence]
                scored = teacher.rerank(
                    question.text, evidence + candidates, top_k=len(evidence) + len(candidates)
                )
                best_evidence = max(chunk.score for chunk in scored if chunk.page in question.evidence)
                keep = {
                    chunk.id
                    for chunk in scored
                    if chunk.page not in question.evidence and chunk.score < best_evidence
                }
                candidates = [chunk for chunk in candidates if chunk.id in keep]
            negatives = tuple(chunk.text for chunk in candidates[:max_negatives])
            examples.append(Example(question.text, positives, negatives))
    return examples
