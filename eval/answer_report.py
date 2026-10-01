"""Writes eval/answer_quality.md: the judge's validation, and the answer prompts compared on dev and test.

Reads results by name from eval/.cache (written by eval.qasper_answers and eval.faithfulness):
    answers_{split}_{prompt}.jsonl                 the random sample of questions
    answers_{split}_unanswerable_{prompt}.jsonl    questions every annotator marked unanswerable
    faithfulness_{split}_{prompt}.jsonl            judge verdicts on the sample's answers
    judge_ragtruth_{judge}.jsonl                   judge verdicts on RAGTruth

Usage:
    python -m eval.answer_report --candidates partial_answers partial_answers_delimited \
        --notes eval/answer_quality_notes.md
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from eval.faithfulness import detection_metrics
from eval.qasper_answers import normalize_answer, refusal_breakdown
from eval.run_qasper import paired_bootstrap

CACHE = Path(__file__).parent / ".cache"
OUTPUT = Path(__file__).parent / "answer_quality.md"
PASSAGES = {"dev": CACHE / "dev_passages.json", "test": CACHE / "kaggle-output" / "answer_passages.json"}
UNANSWERABLE = {
    "dev": CACHE / "dev_unanswerable_passages.json",
    "test": CACHE / "test_unanswerable_passages.json",
}

REPRODUCE = """
Every run below is resumable. Groq's free tier gives each model 200K tokens a day, refilled
continuously; with `--wait-for-quota` a run waits for the budget instead of stopping.

```bash
# Judge validation on RAGTruth
mkdir -p eval/.cache/ragtruth
for f in response source_info; do
  curl -L -o eval/.cache/ragtruth/$f.jsonl \\
    https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/$f.jsonl
done
python -m eval.faithfulness --ragtruth eval/.cache/ragtruth --output eval/.cache/judge_ragtruth_gpt-oss-120b.jsonl

# Passages: the dev sample, and the questions every annotator marked unanswerable
python -m eval.run_qasper --split dev --export-passages 100 --output eval/.cache/dev_passages.json
for split in dev test; do
  python -m eval.run_qasper --split $split --export-passages 60 --only-unanswerable \\
    --output eval/.cache/${{split}}_unanswerable_passages.json
done

# Answers and verdicts for each prompt (the test sample's passages come from the Kaggle job)
for prompt in baseline {candidates}; do
  python -m eval.qasper_answers --passages eval/.cache/dev_passages.json --prompt $prompt \\
    --output eval/.cache/answers_dev_$prompt.jsonl --wait-for-quota
  python -m eval.qasper_answers --passages eval/.cache/dev_unanswerable_passages.json --prompt $prompt \\
    --output eval/.cache/answers_dev_unanswerable_$prompt.jsonl --wait-for-quota
  python -m eval.faithfulness --answers eval/.cache/answers_dev_$prompt.jsonl \\
    --passages eval/.cache/dev_passages.json --output eval/.cache/faithfulness_dev_$prompt.jsonl --wait-for-quota
done
# ...then the same on test for the baseline and the prompt chosen on dev

python -m eval.answer_report --candidates {candidates} --notes eval/answer_quality_notes.md
```
"""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return (centre - half, centre + half)


def bootstrap_mean(values: list[float], resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean with a 95% percentile bootstrap interval."""
    data = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, len(data), size=(resamples, len(data)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(data.mean()), float(low), float(high)


def answerable(record: dict) -> bool:
    return all(ref["type"] != "none" for ref in record["references"])


def yes_no_correct(record: dict) -> bool | None:
    """For yes/no questions with one agreed answer: does the reply start with it?"""
    refs = record["references"]
    if not all(ref["type"] == "boolean" for ref in refs) or len({ref["answer"] for ref in refs}) != 1:
        return None
    return normalize_answer(record["predicted_answer"]).split()[:1] == [refs[0]["answer"].lower()]


def billable_tokens(records: list[dict]) -> float | None:
    usage = [r["usage"] for r in records if r.get("usage")]
    if not usage:
        return None
    return sum(u["prompt_tokens"] - u["cached_tokens"] + u["completion_tokens"] for u in usage) / len(usage)


def prompt_results(split: str, prompt: str, passages: list[dict], unanswerable_total: int) -> dict | None:
    """One prompt's results on a split, or None while its answers to the sample are incomplete."""
    sample = read_jsonl(CACHE / f"answers_{split}_{prompt}.jsonl")
    if not passages or len(sample) < len(passages):
        return None
    answered = {r["question_id"] for r in sample if not r["refused"]}
    verdicts = [v for v in read_jsonl(CACHE / f"faithfulness_{split}_{prompt}.jsonl") if v["id"] in answered]
    return {
        "records": {r["question_id"]: r for r in sample},
        "passages": passages,
        "unanswerable": read_jsonl(CACHE / f"answers_{split}_unanswerable_{prompt}.jsonl"),
        "unanswerable_total": unanswerable_total,
        # Faithfulness only counts once every answered question has a verdict.
        "verdicts": verdicts if len(verdicts) == len(answered) else None,
    }


def interval(result: dict, scale: float = 1.0, unit: str = "", digits: int = 3) -> str:
    low, mean, high = (result[key] * scale for key in ("ci_low", "difference", "ci_high"))
    return f"{mean:+.{digits}f}{unit} ({low:+.{digits}f} to {high:+.{digits}f})"


def values_table(split: str, runs: dict[str, dict]) -> list[str]:
    """Every prompt's scores on the split, one column per prompt."""

    def mean_of(run: dict, field: str) -> str:
        records = run["records"].values()
        return f"{sum(r[field] for r in records) / len(records):.3f}"

    def refused_answerable(run: dict) -> str:
        breakdown = refusal_breakdown(list(run["records"].values()), run["passages"])
        return f"{breakdown['refused']}/{breakdown['answerable']}"

    def refused_with_evidence(run: dict) -> str:
        breakdown = refusal_breakdown(list(run["records"].values()), run["passages"])
        return str(breakdown["refused_with_evidence_in_context"])

    def refused_unanswerable(run: dict) -> str:
        records, total = run["unanswerable"], run["unanswerable_total"]
        if total == 0:
            return "not run"
        if len(records) < total:
            return f"pending ({len(records)} of {total})"
        return f"{sum(r['refused'] for r in records)}/{total}"

    def yes_no(run: dict) -> str:
        correct = [c for c in map(yes_no_correct, run["records"].values()) if c is not None]
        return f"{sum(correct)}/{len(correct)}"

    def faithfulness(run: dict) -> str:
        if run["verdicts"] is None:
            return "pending"
        judged = [v for v in run["verdicts"] if v["claim_count"]]
        mean, low, high = bootstrap_mean([v["faithfulness"] for v in judged])
        return f"{mean:.1%} ({low:.1%} to {high:.1%})"

    def fully_faithful(run: dict) -> str:
        if run["verdicts"] is None:
            return "pending"
        judged = [v for v in run["verdicts"] if v["claim_count"]]
        return f"{sum(not v['flagged'] for v in judged)}/{len(judged)}"

    def tokens(run: dict) -> str:
        value = billable_tokens(list(run["records"].values()))
        return f"{value:.0f}" if value else "not recorded"

    rows = [
        ("Answer F1 (official)", lambda run: mean_of(run, "answer_f1")),
        ("Evidence F1 (official)", lambda run: mean_of(run, "evidence_f1")),
        ("Refused, answerable questions", refused_answerable),
        ("… with an evidence passage in the prompt", refused_with_evidence),
        ("Refused, unanswerable questions", refused_unanswerable),
        ("Yes/no questions answered correctly", yes_no),
        ("Faithfulness: supported share of claims, mean (95% CI)", faithfulness),
        ("Answers with every claim supported", fully_faithful),
        ("Billable tokens per answer", tokens),
    ]
    table = [
        f"| {split.capitalize()} | " + " | ".join(f"`{name}`" for name in runs) + " |",
        "|---|" + "---:|" * len(runs),
    ]
    return table + [
        f"| {label} | " + " | ".join(cell(run) for run in runs.values()) + " |" for label, cell in rows
    ]


def differences_table(runs: dict[str, dict]) -> list[str]:
    """Each candidate minus the baseline on the same questions, with paired bootstrap 95% intervals."""
    baseline = runs["baseline"]
    candidates = {name: run for name, run in runs.items() if name != "baseline"}
    ids = sorted(baseline["records"])
    answerable_ids = [i for i in ids if answerable(baseline["records"][i])]

    def paired(candidate: dict, field: str, subset: list[str], **style) -> str:
        values = [{i: float(run["records"][i][field]) for i in subset} for run in (baseline, candidate)]
        return interval(paired_bootstrap(*values, metric=float), **style)

    rows = [
        ("Answer F1", lambda c: paired(c, "answer_f1", ids)),
        ("Evidence F1", lambda c: paired(c, "evidence_f1", ids)),
        (
            "Refused, answerable questions",
            lambda c: paired(c, "refused", answerable_ids, scale=100, unit=" pts", digits=1),
        ),
    ]
    table = [
        "| Minus `baseline` | " + " | ".join(f"`{name}`" for name in candidates) + " |",
        "|---|" + "---:|" * len(candidates),
    ]
    return table + [
        f"| {label} | " + " | ".join(cell(c) for c in candidates.values()) + " |" for label, cell in rows
    ]


def judge_section(path: Path) -> list[str]:
    records = read_jsonl(path)
    if not records:
        return ["Judge validation has not been run yet."]
    m = detection_metrics(records)
    tp, fp, fn, tn = (
        m[k] for k in ("true_positives", "false_positives", "false_negatives", "true_negatives")
    )
    precision, recall, accuracy = wilson(tp, tp + fp), wilson(tp, tp + fn), wilson(tp + tn, len(records))
    return [
        f"The judge, `{records[0]['judge']}` with low reasoning effort and schema-constrained JSON output, "
        "splits each answer into atomic claims and labels every claim supported, unsupported or contradicted "
        "by the passages the answer was written from. It was first checked against human labels from "
        "[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (Niu et al., ACL 2024; MIT licence): "
        f"{len(records)} question-answering responses from its test split, half of them containing at least "
        "one span annotators marked as hallucinated, refusals excluded. A response counts as flagged when "
        "the judge finds an unsupported or contradicted claim.",
        "",
        "| Agreement with human labels | Value | 95% CI |",
        "|---|---:|---:|",
        f"| Precision | {m['precision']:.2f} | {precision[0]:.2f} to {precision[1]:.2f} |",
        f"| Recall | {m['recall']:.2f} | {recall[0]:.2f} to {recall[1]:.2f} |",
        f"| F1 | {m['f1']:.2f} | |",
        f"| Accuracy | {m['accuracy']:.2f} | {accuracy[0]:.2f} to {accuracy[1]:.2f} |",
        "",
        f"Of the hallucinated responses, {tp} were flagged and {fn} missed; of the clean ones, {fp} were flagged "
        f"and {tn} passed.",
    ]


def split_section(split: str, candidates: list[str]) -> list[str]:
    passages = read_json(PASSAGES[split])
    unanswerable_total = len(read_json(UNANSWERABLE[split]))
    runs = {
        name: result
        for name in ["baseline", *candidates]
        if (result := prompt_results(split, name, passages, unanswerable_total)) is not None
    }
    if "baseline" not in runs or len(runs) == 1:
        return ["Not run yet."]
    missing = [name for name in candidates if name not in runs]
    lines = [
        f"{len(passages)} random {split} questions (answerable or not), and separately {unanswerable_total} {split} "
        "questions that every annotator marked unanswerable, to check that fewer refusals do not mean "
        "answering what the paper does not say. Faithfulness is judged on the questions each prompt answered.",
        "",
        *values_table(split, runs),
        "",
        "Differences on the same questions (paired bootstrap 95% intervals):",
        "",
        *differences_table(runs),
    ]
    if missing:
        lines += ["", f"Not run on {split}: " + ", ".join(f"`{name}`" for name in missing) + "."]
    return lines


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    arguments.add_argument(
        "--candidates", nargs="+", required=True, help="prompts compared with the baseline"
    )
    arguments.add_argument("--judge", default="gpt-oss-120b")
    arguments.add_argument("--notes", type=Path, help="Markdown inserted after the judge section")
    arguments.add_argument("--output", type=Path, default=OUTPUT)
    args = arguments.parse_args()

    lines = [
        "# Answer quality: refusals and faithfulness",
        "",
        "The baseline prompt tells the model to answer only from the retrieved passages and otherwise to "
        "reply with a fixed refusal. On QASPER it refused many answerable questions, so revised prompts "
        "were designed from an error analysis of the dev split and compared with the baseline, first on dev "
        "and then, once, on test. Answers come from `openai/gpt-oss-20b` on Groq, using the four passages "
        "the application's retrieval returns (hybrid search with the off-the-shelf reranker). The prompts "
        "are defined in [prompts.py](prompts.py).",
        "",
        "## The faithfulness judge",
        "",
        *judge_section(CACHE / f"judge_ragtruth_{args.judge}.jsonl"),
    ]
    if args.notes:
        lines += ["", args.notes.read_text().strip()]
    for split in ("dev", "test"):
        lines += ["", f"## {split.capitalize()} split", "", *split_section(split, args.candidates)]
    lines += ["", "## Reproduce", "", REPRODUCE.format(candidates=" ".join(args.candidates)).strip()]
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
