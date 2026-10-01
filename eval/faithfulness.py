"""Faithfulness of generated answers to their sources, scored by an LLM judge.

For each answer that is not a refusal, the judge splits the answer into atomic claims and
labels each one against the numbered sources the generator saw:
  * supported: a source states it, or it follows from the sources without new facts;
  * unsupported: no source states it, even if it may be true;
  * contradicted: a source states something incompatible with it.
An answer's faithfulness is its share of supported claims; it is faithful when every claim is.

Before its scores are trusted, the judge is checked against human hallucination labels from
RAGTruth (Niu et al., ACL 2024; MIT licence): `--ragtruth` judges a balanced sample of
its question-answering responses and reports how well "any unsupported or contradicted
claim" matches "annotators marked a hallucinated span".

Usage:
    python -m eval.faithfulness --ragtruth eval/.cache/ragtruth --output eval/.cache/judge_ragtruth.jsonl
    python -m eval.faithfulness --answers eval/.cache/answers_dev_baseline.jsonl \
        --passages eval/.cache/dev_passages.json --output eval/.cache/faithfulness_dev_baseline.jsonl
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from groq import Groq

from app.config import get_settings
from eval.qasper_answers import call_with_retries, usage_dict

LABELS = ("supported", "unsupported", "contradicted")

JUDGE_SYSTEM_PROMPT = """You check whether an answer is supported by the numbered sources it was written from.

Work in two steps.

1. Split the answer into atomic claims: short, self-contained factual statements that can each be checked on their own. Replace pronouns with what they refer to. Keep a claim's qualifiers (numbers, names, conditions, comparisons) inside that claim. Skip citation markers such as [1], and skip statements that only describe what the sources do or do not contain, such as "The sources do not mention the optimizer".

2. Label each claim using the sources only, never your own knowledge:
   - "supported": a source states it, or it follows from the sources without adding facts (paraphrase, simple arithmetic, or combining statements from several sources counts as supported).
   - "unsupported": no source states it, even if it may well be true. Added details are unsupported: a number, name, method, reason or qualifier that the sources do not give.
   - "contradicted": a source states something incompatible with it.
   For "supported" and "contradicted" claims, list the numbers of the sources that decide the label; for "unsupported" claims, list none.

Be strict about specifics and lenient about wording: a faithful paraphrase is supported, but a claim that adds one unsupported specific is unsupported as a whole. If the answer makes no factual claims, return an empty list.

Example A
Sources:
[1] We train the tagger on 12,000 sentences from the Penn Treebank and evaluate it on section 23.
[2] The tagger reaches 97.1% accuracy, 0.4 points above the CRF baseline.
Question: How well does the tagger perform?
Answer: The tagger reaches 97.1% accuracy on section 23 of the Penn Treebank, beating the CRF baseline by 0.4 points [2]. It was trained on 12,000 sentences with a BiLSTM encoder [1].
Claims:
{"claims": [
 {"claim": "The tagger reaches 97.1% accuracy.", "sources": [2], "label": "supported"},
 {"claim": "The tagger's accuracy is measured on section 23 of the Penn Treebank.", "sources": [1, 2], "label": "supported"},
 {"claim": "The tagger beats the CRF baseline by 0.4 points.", "sources": [2], "label": "supported"},
 {"claim": "The tagger was trained on 12,000 sentences.", "sources": [1], "label": "supported"},
 {"claim": "The tagger uses a BiLSTM encoder.", "sources": [], "label": "unsupported"}
]}

Example B
Sources:
[1] Annotation took three weeks; five annotators labelled every tweet.
[2] Tweets were collected between March and May 2019.
Question: Who annotated the data?
Answer: Four annotators labelled the tweets over three weeks [1]. The tweets were collected in spring 2019 [2].
Claims:
{"claims": [
 {"claim": "Four annotators labelled the tweets.", "sources": [1], "label": "contradicted"},
 {"claim": "Labelling the tweets took three weeks.", "sources": [1], "label": "supported"},
 {"claim": "The tweets were collected in spring 2019.", "sources": [2], "label": "supported"}
]}

Example C
Sources:
[1] We train for 10 epochs with a batch size of 32.
Question: Which optimizer do they use?
Answer: The sources do not say which optimizer was used. They only report that the model was trained for 10 epochs [1].
Claims:
{"claims": [
 {"claim": "The model was trained for 10 epochs.", "sources": [1], "label": "supported"}
]}

Reply with the JSON object only."""

CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "integer"}},
                    "label": {"type": "string", "enum": list(LABELS)},
                },
                "required": ["claim", "sources", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


def format_sources(passages: list[str]) -> str:
    return "\n\n".join(f"[{number}] {text}" for number, text in enumerate(passages, start=1))


def parse_claims(content: str) -> list[dict]:
    """Validates the judge's JSON; raises ValueError on anything malformed."""
    try:
        claims = json.loads(content)["claims"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"judge output is not the expected JSON: {content[:200]!r}") from error
    if not isinstance(claims, list):
        raise ValueError("'claims' is not a list")
    parsed = []
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("label") not in LABELS:
            raise ValueError(f"malformed claim: {claim!r}")
        sources = [s for s in claim.get("sources", []) if isinstance(s, int)]
        parsed.append({"claim": str(claim.get("claim", "")), "sources": sources, "label": claim["label"]})
    return parsed


class Judge:
    """Claim-level faithfulness judge on a Groq chat model with strict JSON output."""

    def __init__(
        self,
        model: str,
        client: Any,
        reasoning_effort: str | None = "low",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.last_usage: Any = None

    def claims(self, question: str, passages: list[str], answer: str) -> list[dict]:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Sources:\n{format_sources(passages)}\nQuestion: {question}\nAnswer: {answer}",
            },
        ]
        options = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        for _ in range(2):  # strict decoding guarantees the schema; a retry covers truncation
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "claims", "strict": True, "schema": CLAIMS_SCHEMA},
                },
                **options,
            )
            self.last_usage = getattr(response, "usage", None)
            try:
                return parse_claims(response.choices[0].message.content or "")
            except ValueError:
                continue
        raise ValueError("the judge returned malformed output twice")


def answer_scores(claims: list[dict]) -> dict:
    counts = Counter(claim["label"] for claim in claims)
    return {
        "claim_count": len(claims),
        **{label: counts[label] for label in LABELS},
        "faithfulness": counts["supported"] / len(claims) if claims else None,
        "flagged": counts["unsupported"] + counts["contradicted"] > 0,
    }


def summarize(records: list[dict]) -> dict:
    """Aggregates over answers with at least one claim (refusals are not judged)."""
    judged = [r for r in records if r["claim_count"]]
    claims = sum(r["claim_count"] for r in judged)
    return {
        "answers_judged": len(judged),
        "answers_without_claims": len(records) - len(judged),
        "mean_faithfulness": sum(r["faithfulness"] for r in judged) / len(judged) if judged else None,
        "fully_faithful_answers": sum(not r["flagged"] for r in judged) / len(judged) if judged else None,
        "claims": claims,
        **{f"{label}_claims": sum(r[label] for r in judged) for label in LABELS},
    }


def detection_metrics(records: list[dict]) -> dict:
    """Judge flags vs human hallucination labels (positive class: hallucinated response)."""
    tp = sum(r["hallucinated"] and r["flagged"] for r in records)
    fp = sum(not r["hallucinated"] and r["flagged"] for r in records)
    fn = sum(r["hallucinated"] and not r["flagged"] for r in records)
    tn = sum(not r["hallucinated"] and not r["flagged"] for r in records)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "responses": len(records),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(records) if records else 0.0,
    }


_PASSAGE_MARKER = re.compile(r"passage \d+:", re.IGNORECASE)


def split_ragtruth_passages(text: str) -> list[str]:
    """RAGTruth stores a question's passages as one string, each starting "passage N:"."""
    parts = [part.strip() for part in _PASSAGE_MARKER.split(text)]
    return [part for part in parts if part] or [text.strip()]


RAGTRUTH_REFUSAL = "unable to answer based on given passages"


def load_ragtruth(directory: Path, per_class: int, seed: int = 0) -> list[dict]:
    """A balanced sample of RAGTruth's test-split QA responses (hallucinated vs not).

    Refusals are left out: they make no claims, so they would be free true negatives.
    """
    with (directory / "source_info.jsonl").open() as lines:
        sources = {source["source_id"]: source for source in map(json.loads, lines)}
    with (directory / "response.jsonl").open() as lines:
        responses = [json.loads(line) for line in lines]
    pool = [
        r
        for r in responses
        if sources[r["source_id"]]["task_type"] == "QA"
        and r["split"] == "test"
        and r["quality"] == "good"
        and RAGTRUTH_REFUSAL not in r["response"].lower()
    ]
    rng = random.Random(seed)
    hallucinated = rng.sample([r for r in pool if r["labels"]], per_class)
    clean = rng.sample([r for r in pool if not r["labels"]], per_class)
    return [
        {
            "id": r["id"],
            "model": r["model"],
            "question": sources[r["source_id"]]["source_info"]["question"],
            "passages": split_ragtruth_passages(sources[r["source_id"]]["source_info"]["passages"]),
            "answer": r["response"],
            "hallucinated": bool(r["labels"]),
        }
        for r in hallucinated + clean
    ]


def qasper_source(passage: dict) -> str:
    """A passage exactly as eval.qasper_answers showed it to the generator (after its "[n] ")."""
    return f"paper, page {passage['paragraph'] + 1}\n{passage['text']}"


def load_qasper_answers(answers: Path, passages: Path) -> list[dict]:
    """Answers from eval.qasper_answers, with the sources the generator saw."""
    by_id = {item["question_id"]: item for item in json.loads(passages.read_text())}
    items = []
    for line in answers.read_text().splitlines():
        record = json.loads(line)
        if record["refused"]:
            continue
        items.append(
            {
                "id": record["question_id"],
                "question": record["question"],
                "passages": [qasper_source(p) for p in by_id[record["question_id"]]["passages"]],
                "answer": record["answer"],
            }
        )
    return items


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = arguments.add_mutually_exclusive_group(required=True)
    source.add_argument("--ragtruth", type=Path, help="directory with RAGTruth's response/source_info.jsonl")
    source.add_argument("--answers", type=Path, help="answers written by eval.qasper_answers")
    arguments.add_argument("--passages", type=Path, help="the passages file those answers came from")
    arguments.add_argument("--per-class", type=int, default=40, help="RAGTruth responses per label")
    arguments.add_argument("--output", type=Path, required=True)
    arguments.add_argument("--model", default="openai/gpt-oss-120b")
    arguments.add_argument("--reasoning-effort", default="low")
    arguments.add_argument("--pace-seconds", type=float, default=6.0)
    arguments.add_argument(
        "--wait-for-quota",
        action="store_true",
        help="wait for Groq's daily budget to refill instead of stopping",
    )
    args = arguments.parse_args()
    if args.answers and not args.passages:
        arguments.error("--answers needs --passages")

    settings = get_settings()
    if settings.groq_api_key is None:
        raise SystemExit("Set GROQ_API_KEY to run the judge.")
    judge = Judge(
        args.model,
        Groq(api_key=settings.groq_api_key.get_secret_value()),
        reasoning_effort=args.reasoning_effort or None,
    )
    if args.ragtruth:
        items = load_ragtruth(args.ragtruth, args.per_class)
    else:
        items = load_qasper_answers(args.answers, args.passages)

    done = {}
    if args.output.exists():  # resume instead of paying for verdicts twice
        for line in args.output.read_text().splitlines():
            record = json.loads(line)
            if record["judge"] != args.model:
                raise SystemExit(f"{args.output} holds verdicts from {record['judge']}; use a new --output")
            done[record["id"]] = record
    with args.output.open("a") as out:
        for item in items:
            if item["id"] in done:
                continue
            claims = call_with_retries(
                judge.claims,
                item["question"],
                item["passages"],
                item["answer"],
                wait_for_quota=args.wait_for_quota,
            )
            record = {
                **{k: v for k, v in item.items() if k != "passages"},
                "judge": args.model,
                "usage": usage_dict(judge.last_usage),
                "claims": claims,
                **answer_scores(claims),
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[item["id"]] = record
            print(f"{len(done)}/{len(items)} flagged={record['flagged']} claims={len(claims)}", flush=True)
            time.sleep(args.pace_seconds)

    records = [done[item["id"]] for item in items]
    summary = detection_metrics(records) if args.ragtruth else summarize(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
