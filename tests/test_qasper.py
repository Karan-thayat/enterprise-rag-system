from dataclasses import replace

import pytest

from app.llm_generator import NOT_FOUND_ANSWER
from eval.hard_negatives import mine_examples
from eval.qasper import first_relevant_rank, index_papers, parse_papers, retrieval_metrics
from eval.qasper_answers import (
    paragraph_f1,
    refusal_breakdown,
    score,
    summarize,
    to_prediction,
    token_f1,
    token_recall,
)
from eval.run_qasper import (
    evaluate,
    export_passages,
    hybrid_ranker,
    paired_bootstrap,
    reciprocal_rank,
)
from eval.train_reranker import mine_pairs


def annotation(evidence, spans=(), free_form="", yes_no=None, unanswerable=False):
    return {
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": list(spans),
            "yes_no": yes_no,
            "free_form_answer": free_form,
            "evidence": list(evidence),
            "highlighted_evidence": [],
        }
    }


RAW = {
    "paper-1": {
        "title": "A tiny paper",
        "abstract": "",
        "full_text": [
            {
                "section_name": "Intro",
                "paragraphs": [
                    "The Eiffel Tower is located in Paris.",
                    "   ",
                    "Photosynthesis happens in leaves.",
                ],
            },
            {"section_name": "Method", "paragraphs": ["Python was created by Guido van Rossum."]},
        ],
        "figures_and_tables": [{"file": "t1.png", "caption": "Table 1: Tower heights."}],
        "qas": [
            {
                "question": "Where is the Eiffel Tower? ",
                "question_id": "q1",
                "answers": [
                    # Double space and a section-heading "evidence" string, as in the real data.
                    annotation(
                        ["The Eiffel Tower is located in  Paris.", "Intro ::: Heading"], spans=["Paris"]
                    ),
                    annotation(["FLOAT SELECTED: Table 1: Tower heights."], free_form="In Paris"),
                ],
            },
            {
                "question": "Who funded it?",
                "question_id": "q2",
                "answers": [annotation([], unanswerable=True)],
            },
            {
                "question": "Who created Python?",
                "question_id": "q3",
                "answers": [annotation(["Python was created by Guido van Rossum."], yes_no=True)],
            },
        ],
    }
}


@pytest.fixture
def paper():
    return parse_papers(RAW)[0]


def test_parse_matches_evidence_to_paragraphs(paper):
    assert paper.paragraphs == (
        "The Eiffel Tower is located in Paris.",
        "Photosynthesis happens in leaves.",
        "Python was created by Guido van Rossum.",
        "FLOAT SELECTED: Table 1: Tower heights.",
    )
    q1, q2, q3 = paper.questions
    assert q1.text == "Where is the Eiffel Tower?"
    assert q1.evidence == {0, 3}  # whitespace-normalised match; the heading is ignored
    assert q2.evidence == frozenset()
    assert q3.evidence == {2}
    assert paper.paragraph_text(3) == "Table 1: Tower heights."


def test_references_follow_the_official_evaluator(paper):
    q1, q2, q3 = paper.questions
    assert [(r["answer"], r["type"]) for r in q1.references] == [
        ("Paris", "extractive"),
        ("In Paris", "abstractive"),
    ]
    assert q2.references == ({"answer": "Unanswerable", "evidence": [], "type": "none"},)
    assert [(r["answer"], r["type"]) for r in q3.references] == [("Yes", "boolean")]


def test_retrieval_metrics_math():
    metrics = retrieval_metrics([1, 3, None, 12])
    assert metrics == pytest.approx({"Hit@1": 0.25, "Hit@4": 0.5, "Hit@10": 0.5, "MRR@10": (1 + 1 / 3) / 4})


def test_evaluate_scores_only_questions_with_evidence(paper, embedder):
    indexed = index_papers([paper], embedder, max_tokens=20, overlap_tokens=4)
    result = evaluate(indexed, hybrid_ranker(embedder))
    assert result["questions"] == 2
    assert set(result["ranks"]) == {"q1", "q3"}
    assert result["ranks"]["q3"] == 1  # the only chunk mentioning Python


def test_first_relevant_rank(paper, embedder):
    indexed = index_papers([paper], embedder)
    chunks = indexed[0].store.all_chunks()
    assert first_relevant_rank(chunks, frozenset({2})) == 3
    assert first_relevant_rank(chunks, frozenset()) is None


def test_paired_bootstrap_brackets_the_mean_difference():
    baseline = {"a": 2, "b": None, "c": 1}
    candidate = {"a": 1, "b": 1, "c": 1}
    result = paired_bootstrap(baseline, candidate, reciprocal_rank, resamples=2000)
    assert result["difference"] == pytest.approx(0.5)
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]
    assert result["ci_low"] >= 0 and result["ci_high"] <= 1


def test_mine_pairs_labels_evidence_chunks_positive_and_the_rest_negative(paper, embedder):
    indexed = index_papers([paper], embedder, max_tokens=20, overlap_tokens=4)
    pairs = mine_pairs(indexed, embedder)
    by_question = {}
    for pair in pairs:
        by_question.setdefault(pair.question, []).append(pair)
    assert set(by_question) == {"Where is the Eiffel Tower?", "Who created Python?"}  # q2 has no evidence
    python_pairs = by_question["Who created Python?"]
    assert [p.passage for p in python_pairs if p.label == 1] == ["Python was created by Guido van Rossum."]
    assert all("Python" not in p.passage for p in python_pairs if p.label == 0)
    assert len(python_pairs) == 4  # one positive, and the three other chunks as negatives


def test_export_passages_keeps_official_evidence_strings(paper, embedder):
    indexed = index_papers([paper], embedder)
    exported = export_passages(indexed, hybrid_ranker(embedder), sample_size=3)
    assert {item["question_id"] for item in exported} == {"q1", "q2", "q3"}
    passages = [p for item in exported for p in item["passages"]]
    assert "FLOAT SELECTED: Table 1: Tower heights." in {p["evidence_string"] for p in passages}
    assert all(len(item["passages"]) <= 4 for item in exported)


def test_answer_scoring_matches_the_official_definitions():
    assert token_f1("the Paris", "Paris.") == 1.0
    assert token_f1("in Paris France", "Paris") == pytest.approx(0.5)
    assert paragraph_f1([], []) == 1.0
    assert paragraph_f1(["a", "b"], ["a"]) == pytest.approx(2 / 3)

    passages = [{"evidence_string": "p1"}, {"evidence_string": "p2"}]
    assert to_prediction(NOT_FOUND_ANSWER, passages) == ("Unanswerable", [])
    assert to_prediction("It is **Paris** [2][2].", passages) == ("It is Paris .", ["p2"])

    references = [{"answer": "Paris", "evidence": ["p2"], "type": "extractive"}]
    assert score("Paris", ["p2"], references) == {
        "answer_f1": 1.0,
        "answer_type": "extractive",
        "evidence_f1": 1.0,
    }


class OpinionatedTeacher:
    """Stands in for a cross-encoder that ranks one unmarked paragraph above the evidence."""

    def rerank(self, query, chunks, top_k):
        def score(chunk):
            return 5.0 if "Photosynthesis" in chunk.text else 1.0 if "Eiffel" in chunk.text else 0.0

        return sorted((replace(chunk, score=score(chunk)) for chunk in chunks), key=lambda c: -c.score)[
            :top_k
        ]


def test_mine_examples_denoising_drops_negatives_the_teacher_ranks_above_the_evidence(paper, embedder):
    indexed = index_papers([paper], embedder, max_tokens=20, overlap_tokens=4)
    raw = {example.question: example for example in mine_examples(indexed, embedder)}
    denoised = {
        example.question: example
        for example in mine_examples(indexed, embedder, teacher=OpinionatedTeacher())
    }
    eiffel = "Where is the Eiffel Tower?"
    assert set(raw) == set(denoised) == {eiffel, "Who created Python?"}
    assert set(raw[eiffel].positives) == {"The Eiffel Tower is located in Paris.", "Table 1: Tower heights."}
    assert "Photosynthesis happens in leaves." in raw[eiffel].negatives
    assert "Photosynthesis happens in leaves." not in denoised[eiffel].negatives
    assert "Python was created by Guido van Rossum." in denoised[eiffel].negatives


def test_supplementary_metrics_separate_verbosity_from_errors():
    assert token_recall("The dataset contains 829 instances [1]", "829 instances") == 1.0
    records = [
        {"predicted_answer": "Yes. It does", "references": [{"answer": "No", "type": "boolean"}] * 2},
        {"predicted_answer": "No, they do not", "references": [{"answer": "No", "type": "boolean"}]},
        {
            "predicted_answer": "It has 829 instances",
            "references": [{"answer": "829 instances", "type": "extractive"}],
        },
    ]
    for record in records:
        record.update(
            answer_f1=0.0, answer_type=record["references"][0]["type"], evidence_f1=0.0, refused=False
        )
    summary = summarize(records)
    assert summary["boolean_accuracy"] == "1/2"
    assert summary["answer_token_recall"] == pytest.approx(2 / 3)


def test_refusal_breakdown_separates_retrieval_misses_from_declines():
    references = [{"answer": "Paris", "evidence": ["gold paragraph"], "type": "extractive"}]
    passages = [
        {"question_id": "hit", "references": references, "passages": [{"evidence_string": "gold paragraph"}]},
        {
            "question_id": "miss",
            "references": references,
            "passages": [{"evidence_string": "other paragraph"}],
        },
        {"question_id": "ok", "references": references, "passages": [{"evidence_string": "gold paragraph"}]},
    ]
    records = [
        {"question_id": "hit", "refused": True, "references": references},
        {"question_id": "miss", "refused": True, "references": references},
        {"question_id": "ok", "refused": False, "references": references},
    ]
    assert refusal_breakdown(records, passages) == {
        "answerable": 3,
        "refused": 2,
        "refused_with_evidence_in_context": 1,
        "refused_after_retrieval_miss": 1,
    }
