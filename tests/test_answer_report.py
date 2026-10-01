import pytest

from eval.answer_report import (
    billable_tokens,
    bootstrap_mean,
    differences_table,
    values_table,
    wilson,
    yes_no_correct,
)


def record(question_id, f1, refused=False, kind="extractive", answer="Paris", reference="Paris"):
    return {
        "question_id": question_id,
        "answer_f1": f1,
        "evidence_f1": f1,
        "refused": refused,
        "predicted_answer": "Unanswerable" if refused else answer,
        "references": [{"type": kind, "answer": reference, "evidence": []}],
        "usage": {"prompt_tokens": 600, "cached_tokens": 100, "completion_tokens": 200},
    }


def test_wilson_interval_matches_the_textbook_values():
    low, high = wilson(32, 40)
    assert low == pytest.approx(0.652, abs=1e-3) and high == pytest.approx(0.895, abs=1e-3)
    assert wilson(0, 0) == (0.0, 1.0)


def test_bootstrap_mean_brackets_the_mean():
    mean, low, high = bootstrap_mean([0.0, 0.5, 1.0, 1.0])
    assert mean == 0.625 and low <= mean <= high


def test_yes_no_scoring_needs_one_agreed_answer():
    assert yes_no_correct(record("a", 1.0, kind="boolean", answer="Yes, because", reference="Yes")) is True
    assert yes_no_correct(record("b", 0.0, kind="boolean", answer="No.", reference="Yes")) is False
    assert yes_no_correct(record("c", 1.0)) is None


def test_billable_tokens_exclude_cached_prompt_tokens():
    assert billable_tokens([record("a", 1.0)]) == 700
    assert billable_tokens([{"usage": None}]) is None


def test_tables_count_refusals_on_answerable_questions_only():
    baseline_records = [
        record("a", 0.0, refused=True),
        record("b", 1.0),
        record("u", 1.0, refused=True, kind="none"),
    ]
    candidate_records = [record("a", 1.0), record("b", 1.0), record("u", 1.0, refused=True, kind="none")]
    passages = [
        {"question_id": r["question_id"], "references": r["references"], "passages": []}
        for r in baseline_records
    ]

    def run(records, unanswerable=()):
        return {
            "records": {r["question_id"]: r for r in records},
            "passages": passages,
            "unanswerable": list(unanswerable),
            "unanswerable_total": 2,
            "verdicts": None,
        }

    runs = {
        "baseline": run(baseline_records),
        "candidate": run(candidate_records, [record("x", 1.0, refused=True)]),
    }
    values = "\n".join(values_table("dev", runs))
    assert "| Refused, answerable questions | 1/2 | 0/2 |" in values
    assert "| Refused, unanswerable questions | pending (0 of 2) | pending (1 of 2) |" in values
    assert "| Faithfulness: supported share of claims, mean (95% CI) | pending | pending |" in values
    differences = "\n".join(differences_table(runs))
    assert "| Refused, answerable questions | -50.0 pts" in differences
