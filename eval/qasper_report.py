"""Writes eval/qasper_results.md from the Kaggle job's results.json and the local answer scores.

Usage:
    python -m eval.qasper_report --results eval/.cache/kaggle-output/results.json \
        --answers eval/.cache/qasper_answers.jsonl
"""

import argparse
import json
from pathlib import Path

from eval.qasper_answers import refusal_breakdown, summarize

OUTPUT = Path(__file__).parent / "qasper_results.md"


def pct(value: float) -> str:
    return f"{value:.1%}"


def retrieval_table(results: dict) -> str:
    lines = ["| Retrieval pipeline | Hit@1 | Hit@4 | Hit@10 | MRR@10 |", "|---|---:|---:|---:|---:|"]
    for name, row in results.items():
        lines.append(
            f"| {name} | {pct(row['Hit@1'])} | {pct(row['Hit@4'])} | {pct(row['Hit@10'])} | {row['MRR@10']:.3f} |"
        )
    return "\n".join(lines)


def interval(result: dict, points: bool) -> str:
    if points:
        return f"{result['difference'] * 100:+.1f} pts (95% CI {result['ci_low'] * 100:+.1f} to {result['ci_high'] * 100:+.1f})"
    return f"{result['difference']:+.3f} (95% CI {result['ci_low']:+.3f} to {result['ci_high']:+.3f})"


def round2_section(round2: dict) -> list[str]:
    systems = [name for name in round2["test"] if "candidate recall" not in name]
    recalls = [name for name in round2["test"] if "candidate recall" in name]
    table = [
        "| System | Dev MRR@10 | Test Hit@1 | Test Hit@4 | Test Hit@10 | Test MRR@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in systems:
        test, dev = round2["test"][name], round2["dev"][name]
        mark = " **(selected on dev)**" if name == round2["selected_on_dev"] else ""
        table.append(
            f"| {name}{mark} | {dev['MRR@10']:.3f} | {pct(test['Hit@1'])} | {pct(test['Hit@4'])} | "
            f"{pct(test['Hit@10'])} | {test['MRR@10']:.3f} |"
        )
    recall_rows = [
        f"| {name.split(':')[0]} | {pct(round2['dev'][name]['Hit@20'])} | {pct(round2['test'][name]['Hit@20'])} |"
        for name in recalls
    ]
    runs = ["| Component | Negatives | Epoch | Train loss | Dev MRR@10 |", "|---|---|---:|---:|---:|"]
    for component, rows in round2["training"].items():
        for run in rows:
            for row in run["history"]:
                runs.append(
                    f"| {component} | {run['variant']} | {row['epoch']} | {row['train_loss']:.3f} | {row['dev_score']:.3f} |"
                )
    significance = round2["significance_on_test"]
    selected = significance["selected system vs off-the-shelf"]
    recall_test = significance["fine-tuned vs off-the-shelf embedder, candidate recall@20"]
    return [
        "",
        "## Round 2: fine-tuning the embedding model, and a listwise reranker",
        "",
        "Round 1's reranker lost to the off-the-shelf model, so round 2 changed two things suggested by that result: "
        "a listwise loss (softmax over each question's evidence and its hard negatives) instead of pointwise BCE, and "
        "hard negatives *denoised* by the off-the-shelf reranker, which drops candidates it ranks above every marked "
        "evidence chunk, because QASPER annotators do not mark every supporting paragraph. It also fine-tunes the "
        "embedding model, since the reranker can only reorder the 20 candidates the first stage finds. Each component "
        "was trained with raw and with denoised negatives, and every epoch was scored on dev. Round 2 was designed "
        "after round 1's results, including its test scores, were known; every round-2 choice (variant, epoch and "
        "the final combination) was made on dev.",
        "",
        "Candidate recall@20 (share of questions with evidence among the 20 chunks the reranker sees):",
        "",
        "| Embedder | Dev | Test |",
        "|---|---:|---:|",
        *recall_rows,
        "",
        f"Fine-tuned minus off-the-shelf embedder, candidate recall@20 on test: {interval(recall_test, points=True)}.",
        "",
        "End-to-end retrieval (hybrid + reranking) for every embedder and reranker combination:",
        "",
        *table,
        "",
        f"Selected system minus the off-the-shelf pair on test: MRR@10 {interval(selected['MRR@10'], points=False)}, "
        f"Hit@1 {interval(selected['Hit@1'], points=True)}, Hit@4 {interval(selected['Hit@4'], points=True)}.",
        "",
        "Training runs (every epoch's checkpoint scored on dev):",
        "",
        *runs,
        "",
        f"Run on Kaggle ({round2['environment']['device']}) in {round2['runtime_minutes']} minutes.",
    ]


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    arguments.add_argument("--results", type=Path, required=True)
    arguments.add_argument("--answers", type=Path, default=None)
    arguments.add_argument(
        "--round2", type=Path, default=None, help="results_round2.json from kaggle/round2_job.py"
    )
    arguments.add_argument(
        "--passages", type=Path, default=None, help="answer_passages.json, for the refusal analysis"
    )
    arguments.add_argument("--output", type=Path, default=OUTPUT)
    args = arguments.parse_args()
    results = json.loads(args.results.read_text())
    splits, training, env = results["splits"], results["training"], results["environment"]
    selected = training["selected"]

    grid = ["| Learning rate | Epoch | Train loss | Dev MRR@10 |", "|---:|---:|---:|---:|"]
    for run in training["runs"]:
        for row in run["history"]:
            mark = (
                " (selected)"
                if (run["learning_rate"], row["epoch"]) == (selected["learning_rate"], selected["epoch"])
                else ""
            )
            grid.append(
                f"| {run['learning_rate']:g} | {row['epoch']}{mark} | {row['train_loss']:.3f} | {row['dev_score']:.3f} |"
            )

    significance = ["| Metric | Fine-tuned minus off-the-shelf | 95% CI |", "|---|---:|---:|"]
    for metric, s in results["significance_test_tuned_vs_base"].items():
        scale = (lambda v: f"{v:+.3f}") if metric.startswith("MRR") else (lambda v: f"{v * 100:+.1f} pts")
        significance.append(
            f"| {metric} | {scale(s['difference'])} | {scale(s['ci_low'])} to {scale(s['ci_high'])} |"
        )

    lines = [
        "# QASPER benchmark",
        "",
        "[QASPER](https://huggingface.co/datasets/allenai/qasper) (Dasigi et al., NAACL 2021; CC BY 4.0) has questions "
        "written by NLP practitioners who had read only each paper's title and abstract, answered by others who marked "
        "the supporting paragraphs in the full text. Each "
        "question is answered against its own paper, using the application's retrieval code: 200-token sentence "
        "chunks, all-MiniLM-L6-v2 embeddings, BM25, Reciprocal Rank Fusion and a cross-encoder reranker.",
        "",
        "| Split | Papers | Questions | With evidence paragraphs |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {s['papers']} | {s['questions']} | {s['questions_with_evidence']} |"
            for name, s in splits.items()
        ],
        "",
        "**Protocol.** A retrieved chunk counts as relevant when its paragraph (or figure/table caption) was marked "
        "as evidence by any annotator; only questions with evidence are scored. The reranker was fine-tuned on the "
        "train split, every choice was made on dev, and the test split was evaluated once per round, after that "
        "round's choices were fixed. Hit@4 is what "
        "the application's LLM sees.",
        "",
        f"## Retrieval on the test split ({results['test']['Dense only']['questions']} questions)",
        "",
        retrieval_table(results["test"]),
        "",
        "Fine-tuned against off-the-shelf reranking on the same test questions (paired bootstrap, 10,000 resamples):",
        "",
        *significance,
        "",
        "## Round 1: fine-tuning the reranker",
        "",
        f"Starting from `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained with binary cross-entropy on "
        f"{training['pairs']:,} (question, chunk) pairs from the train split ({training['positive_pairs']:,} positive). "
        "Positives are chunks of evidence paragraphs; negatives are the highest-ranked non-evidence chunks from the "
        "hybrid retriever, which is exactly what the reranker has to sort in the application. Batch 32, linear "
        "warm-up over 10% of steps, "
        + ("fp16 mixed precision on the GPU." if env["device"] != "cpu" else "on CPU."),
        "",
        *grid,
        "",
        f"Run on Kaggle ({env['device']}, torch {env['torch']}, sentence-transformers {env['sentence_transformers']}) "
        f"in {results['runtime_minutes']} minutes, including all evaluations.",
    ]

    if args.answers and args.answers.exists():
        records = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
        summary = summarize(records)
        by_type = summary["answer_f1_by_type"]
        lines += [
            "",
            f"## Answer quality ({summary['questions']} random test questions)",
            "",
            f"Answers from `openai/gpt-oss-20b` on Groq, using the passages retrieved with the "
            f"{results['answer_scoring_reranker']} reranker and the application's own prompt, scored with QASPER's "
            "official metrics (best match over annotators). Unanswerable questions are included, as in the official "
            "evaluation.",
            "",
            "| Metric | Score |",
            "|---|---:|",
            f"| Answer F1 | {summary['answer_f1']:.3f} |",
            f"| Evidence F1 (paragraphs the answer cites) | {summary['evidence_f1']:.3f} |",
            *[
                f"| Answer F1, {'unanswerable' if kind == 'none' else kind} questions | {score:.3f} |"
                for kind, score in by_type.items()
                if score is not None
            ],
            f"| Answer token recall (supplementary: gold answer tokens present in the reply) | {summary['answer_token_recall']:.3f} |",
            f"| Yes/no accuracy (supplementary: reply starts with the agreed answer) | {summary['boolean_accuracy']} |",
            f"| Refused when every annotator marked the question unanswerable | {summary['refused_when_unanswerable']} |",
            f"| Refused when every annotator answered it | {summary['refused_when_answerable']} |",
            "",
            "Answer F1 is token overlap with short reference answers, so full-sentence answers score lower than "
            "extractive systems even when correct; the per-type rows and refusal counts show where the system "
            "actually fails.",
        ]
        if args.passages and args.passages.exists():
            breakdown = refusal_breakdown(records, json.loads(args.passages.read_text()))
            lines += [
                "",
                f"**Where the refusals come from.** Of {breakdown['answerable']} answerable questions, "
                f"{breakdown['refused']} were refused. In {breakdown['refused_after_retrieval_miss']} the evidence was "
                "not among the retrieved passages, so refusing was the grounded choice. In "
                f"{breakdown['refused_with_evidence_in_context']} a passage from an evidence paragraph was in the prompt "
                "(an upper bound, since a long paragraph can span several chunks), which points at an over-cautious "
                "prompt or model rather than retrieval. Tuning the prompt for this belongs on the dev split, not test.",
            ]

    if args.round2 and args.round2.exists():
        lines += round2_section(json.loads(args.round2.read_text()))

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "kaggle/launch.sh                                   # GPU job 1: benchmark + reranker (Kaggle account needed)",
        "kaggle kernels output <user>/enterprise-rag-qasper-reranker -p eval/.cache/kaggle-output",
        "kaggle/launch.sh kaggle/round2_job.py enterprise-rag-qasper-round2   # GPU job 2: second round",
        "kaggle kernels output <user>/enterprise-rag-qasper-round2 -p eval/.cache/kaggle-round2",
        "python -m eval.qasper_answers --passages eval/.cache/kaggle-output/answer_passages.json",
        "python -m eval.qasper_report --results eval/.cache/kaggle-output/results.json "
        "--answers eval/.cache/qasper_answers.jsonl --passages eval/.cache/kaggle-output/answer_passages.json "
        "--round2 eval/.cache/kaggle-round2/results_round2.json",
        "```",
    ]
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
