"""Answer quality on QASPER, scored with the dataset's official metrics.

Reads the passages retrieved on Kaggle (answer_passages.json), has the application's
LLM client answer each question from them, and scores the answers the way QASPER's
official evaluator does:
  * Answer F1: SQuAD token F1 against each annotator's answer, keeping the best match;
    "I couldn't find the answer" counts as the answer "Unanswerable".
  * Evidence F1: overlap between the paragraphs the answer cites and each annotator's
    evidence paragraphs, keeping the best match.

Usage:
    python -m eval.qasper_answers --passages eval/.cache/kaggle-output/answer_passages.json
"""

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import groq

from app.config import get_settings
from app.llm_generator import NOT_FOUND_ANSWER, LLMClient
from app.vector_db import RetrievedChunk

CITATION = re.compile(r"\[(\d+)\]")
PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, articles and extra whitespace (SQuAD v1.1, as in QASPER)."""
    text = "".join(ch for ch in text.lower() if ch not in PUNCTUATION)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def token_f1(prediction: str, reference: str) -> float:
    predicted, gold = normalize_answer(prediction).split(), normalize_answer(reference).split()
    same = sum((Counter(predicted) & Counter(gold)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(predicted), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Share of the reference answer's tokens present in the prediction (supplementary, not official)."""
    predicted, gold = normalize_answer(prediction).split(), normalize_answer(reference).split()
    if not gold:
        return 0.0
    return sum((Counter(predicted) & Counter(gold)).values()) / len(gold)


def paragraph_f1(predicted: list[str], gold: list[str]) -> float:
    if not predicted and not gold:
        return 1.0  # unanswerable question, and no evidence predicted
    same = len(set(gold) & set(predicted))
    if same == 0:
        return 0.0
    precision, recall = same / len(predicted), same / len(gold)
    return 2 * precision * recall / (precision + recall)


def to_prediction(answer: str, passages: list[dict]) -> tuple[str, list[str]]:
    """Maps an application answer to QASPER's (answer text, evidence paragraphs) format."""
    if normalize_answer(NOT_FOUND_ANSWER) in normalize_answer(answer):
        return "Unanswerable", []
    cited = {int(n) for n in CITATION.findall(answer)}
    evidence = [p["evidence_string"] for number, p in enumerate(passages, start=1) if number in cited]
    text = " ".join(CITATION.sub(" ", answer).replace("**", "").split())
    return text, list(dict.fromkeys(evidence))


def score(prediction: str, evidence: list[str], references: list[dict]) -> dict:
    answer_scores = [(token_f1(prediction, ref["answer"]), ref["type"]) for ref in references]
    best_f1, best_type = max(answer_scores, key=lambda pair: pair[0])
    return {
        "answer_f1": best_f1,
        "answer_type": best_type,
        "evidence_f1": max(paragraph_f1(evidence, ref["evidence"]) for ref in references),
    }


def summarize(scored: list[dict]) -> dict:
    mean = lambda values: sum(values) / len(values) if values else None  # noqa: E731
    by_type = {
        kind: mean([s["answer_f1"] for s in scored if s["answer_type"] == kind])
        for kind in ("extractive", "abstractive", "boolean", "none")
    }
    counts = Counter(s["answer_type"] for s in scored)

    # Refusals, on questions where all annotators agree on answerability.
    def agreed(s: dict, unanswerable: bool) -> bool:
        return all((ref["type"] == "none") == unanswerable for ref in s["references"])

    unanswerable = [s for s in scored if agreed(s, True)]
    answerable = [s for s in scored if agreed(s, False)]
    # Yes/no questions where every annotator gave the same answer: does the reply start with it?
    boolean = [
        s
        for s in scored
        if all(ref["type"] == "boolean" for ref in s["references"])
        and len({ref["answer"] for ref in s["references"]}) == 1
    ]
    boolean_correct = sum(
        normalize_answer(s["predicted_answer"]).split()[:1] == [s["references"][0]["answer"].lower()]
        for s in boolean
    )
    return {
        "questions": len(scored),
        "answer_f1": mean([s["answer_f1"] for s in scored]),
        "evidence_f1": mean([s["evidence_f1"] for s in scored]),
        "answer_f1_by_type": by_type,
        "questions_by_best_matching_type": dict(counts),
        "answer_token_recall": mean(
            [
                max(token_recall(s["predicted_answer"], ref["answer"]) for ref in s["references"])
                for s in scored
            ]
        ),
        "boolean_accuracy": f"{boolean_correct}/{len(boolean)}",
        "refused_when_unanswerable": f"{sum(s['refused'] for s in unanswerable)}/{len(unanswerable)}",
        "refused_when_answerable": f"{sum(s['refused'] for s in answerable)}/{len(answerable)}",
    }


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    arguments.add_argument("--passages", type=Path, required=True)
    arguments.add_argument("--output", type=Path, default=Path("eval/.cache/qasper_answers.jsonl"))
    arguments.add_argument("--limit", type=int, default=None)
    arguments.add_argument(
        "--pace-seconds", type=float, default=12.0, help="stay under Groq's free-tier limits"
    )
    args = arguments.parse_args()

    settings = get_settings()
    api_key = settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
    llm = LLMClient(
        settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        reasoning_effort=settings.llm_reasoning_effort,
        api_key=api_key,
    )
    items = json.loads(args.passages.read_text())[: args.limit]

    done = {}
    if args.output.exists():  # resume an interrupted run instead of paying for answers twice
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            done[record["question_id"]] = record
    with args.output.open("a") as out:
        for item in items:
            if item["question_id"] in done:
                continue
            chunks = [
                RetrievedChunk(p["chunk_id"], p["text"], item["paper_id"], "paper", p["paragraph"] + 1)
                for p in item["passages"]
            ]
            for attempt in range(3):
                try:
                    answer = llm.generate_answer(item["question"], chunks, [])
                    break
                except groq.RateLimitError:
                    time.sleep(60 * (attempt + 1))
            else:
                raise SystemExit("Groq kept rate limiting; rerun later to resume.")
            prediction, evidence = to_prediction(answer, item["passages"])
            record = {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": answer,
                "predicted_answer": prediction,
                "predicted_evidence": evidence,
                "refused": prediction == "Unanswerable",
                "references": item["references"],
                **score(prediction, evidence, item["references"]),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[record["question_id"]] = record
            print(
                f"{len(done)}/{len(items)} F1={record['answer_f1']:.2f} {item['question'][:70]}", flush=True
            )
            time.sleep(args.pace_seconds)

    scored = [done[item["question_id"]] for item in items]
    print(json.dumps(summarize(scored), indent=2))


if __name__ == "__main__":
    main()


def refusal_breakdown(records: list[dict], passages: list[dict]) -> dict:
    """Splits refusals on answerable questions into retrieval misses and declines despite evidence.

    "Evidence in context" means a passage came from an annotated evidence paragraph; a long
    paragraph can be split into several chunks, so this is an upper bound on how often the
    answer sentence itself was in the prompt.
    """
    by_id = {item["question_id"]: item for item in passages}

    def evidence_in_context(record: dict) -> bool:
        item = by_id[record["question_id"]]
        gold = {text for ref in item["references"] for text in ref["evidence"]}
        return any(passage["evidence_string"] in gold for passage in item["passages"])

    answerable = [r for r in records if all(ref["type"] != "none" for ref in r["references"])]
    refused = [r for r in answerable if r["refused"]]
    with_evidence = sum(evidence_in_context(r) for r in refused)
    return {
        "answerable": len(answerable),
        "refused": len(refused),
        "refused_with_evidence_in_context": with_evidence,
        "refused_after_retrieval_miss": len(refused) - with_evidence,
    }
