"""Publishes the QASPER fine-tuned models to the Hugging Face Hub, each with its model card.

Log in once with a token that can write (`hf auth login`); it is stored in ~/.cache/huggingface.

Usage:
    python -m eval.publish_models --dry-run     # print the model cards, upload nothing
    python -m eval.publish_models               # public repos under your account
"""

import argparse
from pathlib import Path

ROUND2 = Path("eval/.cache/kaggle-round2")
CARDS = Path(__file__).parent / "model_cards"
MODELS = {
    "embedder": ("all-MiniLM-L6-v2-qasper", ROUND2 / "qasper-embedder", CARDS / "qasper-embedder.md"),
    "reranker": ("ms-marco-MiniLM-L6-v2-qasper", ROUND2 / "qasper-reranker-v2", CARDS / "qasper-reranker.md"),
}


def render_card(kind: str, repo_ids: dict[str, str]) -> str:
    """Fills a model card's placeholders with the repository names it will be published under."""
    return (
        MODELS[kind][2]
        .read_text()
        .replace("{repo_id}", repo_ids[kind])
        .replace("{embedder_url}", f"https://huggingface.co/{repo_ids['embedder']}")
        .replace("{reranker_url}", f"https://huggingface.co/{repo_ids['reranker']}")
    )


def main() -> None:
    arguments = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    arguments.add_argument("--namespace", help="user or organisation (default: the logged-in user)")
    arguments.add_argument("--private", action="store_true")
    arguments.add_argument("--dry-run", action="store_true")
    args = arguments.parse_args()

    from huggingface_hub import HfApi  # only needed here, not by the rest of the evaluation code

    api = HfApi()
    namespace = args.namespace or ("your-username" if args.dry_run else api.whoami()["name"])
    repo_ids = {kind: f"{namespace}/{name}" for kind, (name, _, _) in MODELS.items()}
    for kind, (_, folder, _) in MODELS.items():
        card = render_card(kind, repo_ids)
        if args.dry_run:
            print(card)
            continue
        if not (folder / "model.safetensors").exists():
            raise SystemExit(f"{folder} has no model; download the round-2 Kaggle output first")
        repo_id = repo_ids[kind]
        api.create_repo(repo_id, private=args.private, exist_ok=True)
        # The training run's auto-generated README is replaced by the card written for the Hub.
        api.upload_folder(
            repo_id=repo_id, folder_path=folder, ignore_patterns=["README.md"], commit_message="Add model"
        )
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=card.encode(),
            path_in_repo="README.md",
            commit_message="Add model card",
        )
        print(f"Published https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
