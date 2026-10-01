import json
from types import SimpleNamespace

import pytest

from eval.faithfulness import (
    CLAIMS_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    Judge,
    answer_scores,
    detection_metrics,
    parse_claims,
    split_ragtruth_passages,
    summarize,
)

CLAIMS = {
    "claims": [
        {"claim": "Paris is in France.", "sources": [1], "label": "supported"},
        {"claim": "Paris has 30 million people.", "sources": [], "label": "unsupported"},
    ]
}


class FakeJudgeClient:
    """Mimics ``chat.completions.create`` with canned judge replies and records every call."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def test_judge_sees_numbered_sources_and_must_follow_the_schema():
    client = FakeJudgeClient([json.dumps(CLAIMS)])
    claims = Judge("judge-model", client).claims(
        "Where is Paris?", ["Paris is in France.", "Lyon is in France."], "Paris is in France [1]."
    )

    assert [claim["label"] for claim in claims] == ["supported", "unsupported"]
    (call,) = client.calls
    assert call["messages"][0] == {"role": "system", "content": JUDGE_SYSTEM_PROMPT}
    assert "[1] Paris is in France.\n\n[2] Lyon is in France." in call["messages"][1]["content"]
    assert call["messages"][1]["content"].endswith("Answer: Paris is in France [1].")
    schema = call["response_format"]["json_schema"]
    assert (schema["strict"], schema["schema"]) == (True, CLAIMS_SCHEMA)
    assert call["reasoning_effort"] == "low"


def test_judge_retries_malformed_output_once():
    client = FakeJudgeClient(['{"claims": [', json.dumps({"claims": []})])
    assert Judge("judge-model", client).claims("q", ["s"], "a") == []
    assert len(client.calls) == 2

    with pytest.raises(ValueError):
        Judge("judge-model", FakeJudgeClient(["{", "{"])).claims("q", ["s"], "a")


def test_parse_claims_rejects_labels_outside_the_scale():
    with pytest.raises(ValueError):
        parse_claims(json.dumps({"claims": [{"claim": "x", "sources": [], "label": "plausible"}]}))
    with pytest.raises(ValueError):
        parse_claims(json.dumps({"verdict": "fine"}))


def test_scores_count_unsupported_and_contradicted_claims_against_the_answer():
    mixed = answer_scores(parse_claims(json.dumps(CLAIMS)))
    assert mixed == {
        "claim_count": 2,
        "supported": 1,
        "unsupported": 1,
        "contradicted": 0,
        "faithfulness": 0.5,
        "flagged": True,
    }
    clean = answer_scores([{"claim": "a", "sources": [1], "label": "supported"}])
    no_claims = answer_scores([])
    assert (clean["flagged"], no_claims["faithfulness"]) == (False, None)

    summary = summarize([mixed, clean, no_claims])
    assert (summary["answers_judged"], summary["answers_without_claims"]) == (2, 1)
    assert (summary["mean_faithfulness"], summary["fully_faithful_answers"]) == (0.75, 0.5)
    assert (summary["claims"], summary["unsupported_claims"]) == (3, 1)


def test_detection_metrics_treat_hallucinated_responses_as_positives():
    outcomes = [(True, True), (True, False), (False, True), (False, False), (False, False)]
    metrics = detection_metrics([{"hallucinated": h, "flagged": f} for h, f in outcomes])
    confusion = ("true_positives", "false_positives", "false_negatives", "true_negatives")
    assert tuple(metrics[key] for key in confusion) == (1, 1, 1, 2)
    assert (metrics["precision"], metrics["recall"], metrics["f1"], metrics["accuracy"]) == (
        0.5,
        0.5,
        0.5,
        0.6,
    )


def test_ragtruth_passages_are_split_on_their_markers():
    text = "passage 1:Techs are paid hourly.\n\npassage 2:Some earn commission.\n\npassage 3:Pay varies."
    assert split_ragtruth_passages(text) == ["Techs are paid hourly.", "Some earn commission.", "Pay varies."]
    assert split_ragtruth_passages("No markers at all.") == ["No markers at all."]


def test_the_judge_sees_qasper_sources_exactly_as_the_generator_did():
    from app.llm_generator import format_sources as generator_sources
    from app.vector_db import RetrievedChunk
    from eval.faithfulness import format_sources, qasper_source

    passages = [{"paragraph": 4, "text": "We use LCSTS."}, {"paragraph": 0, "text": "Abstract text."}]
    # eval.qasper_answers builds these chunks: file "paper", page = paragraph id + 1.
    chunks = [
        RetrievedChunk(f"c{i}", p["text"], "paper-id", "paper", p["paragraph"] + 1)
        for i, p in enumerate(passages)
    ]
    assert format_sources([qasper_source(p) for p in passages]) == generator_sources(chunks)
