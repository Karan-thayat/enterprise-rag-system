import re

import yaml

from eval.publish_models import MODELS, render_card

REPO_IDS = {"embedder": "someone/embedder-qasper", "reranker": "someone/reranker-qasper"}


def test_model_cards_render_without_placeholders_and_with_hub_metadata():
    for kind in MODELS:
        card = render_card(kind, REPO_IDS)
        assert not re.search(r"\{(repo_id|embedder_url|reranker_url)\}", card)
        assert REPO_IDS[kind] in card
        metadata = yaml.safe_load(card.split("---")[1])
        assert metadata["license"] == "apache-2.0"
        assert metadata["datasets"] == ["allenai/qasper"]
    assert "https://huggingface.co/someone/embedder-qasper" in render_card("reranker", REPO_IDS)
    assert "https://huggingface.co/someone/reranker-qasper" in render_card("embedder", REPO_IDS)
