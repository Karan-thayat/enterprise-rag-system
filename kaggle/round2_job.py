"""Kaggle GPU job 2: second fine-tuning round (embedding model, and a new reranker recipe).

Round 1 (qasper_job.py) showed that fine-tuning the reranker with a pointwise loss on
raw hard negatives did not beat the off-the-shelf model on dev. This round tries:
  A. The embedding model (first stage), MultipleNegativesRankingLoss with hard
     negatives, raw and denoised by the off-the-shelf reranker; 3 epochs each.
  B. The reranker with a listwise loss, raw and denoised negatives; 5 epochs each.
Every checkpoint is scored on dev (end-to-end MRR@10 of hybrid retrieval + reranking).
The four embedder x reranker combinations are compared on dev, the best one is
selected, and the test split is evaluated once, with paired bootstrap confidence
intervals for the selected system against the off-the-shelf one.

Outputs in /kaggle/working: results_round2.json, qasper-embedder/, qasper-reranker-v2/.
"""

import json
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/rag-job")  # not saved as kernel output
QASPER_URLS = [
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
]
BASE_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
BASE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VARIANTS = {"raw hard negatives": False, "denoised hard negatives": True}


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def extract(archive: Path, target: Path) -> None:
    with tarfile.open(archive) as tar:
        try:
            tar.extractall(target, filter="data")  # safe extraction on Python >= 3.12
        except TypeError:
            tar.extractall(target)


def install_dependencies() -> None:
    packages = ["rank-bm25==0.2.2", "liteparse==1.2.1"]
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        packages.append("sentence-transformers")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


def locate_code() -> Path:
    """Unpacks the code kaggle/launch.sh embeds in this script; older runs attached it as a dataset."""
    embedded = globals().get("EMBEDDED_CODE")
    if embedded:
        import base64

        archive = SCRATCH / "code.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(base64.b64decode(embedded))
        extract(archive, SCRATCH / "code")
        return SCRATCH / "code"
    for marker in Path("/kaggle/input").rglob("pipeline.py"):
        if marker.parent.name == "app":
            return marker.parent.parent
    for archive in Path("/kaggle/input").rglob("code.tar.gz"):
        target = SCRATCH / "code"
        extract(archive, target)
        return target
    raise SystemExit("Project code not found under /kaggle/input")


def download_qasper(target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for url in QASPER_URLS:
        archive = target / url.rsplit("/", 1)[1]
        urllib.request.urlretrieve(url, archive)
        extract(archive, target)
    return target


def main() -> None:
    started = time.time()
    install_dependencies()
    sys.path.insert(0, str(locate_code()))

    import sentence_transformers
    import torch
    import transformers

    from app.embeddings import Embedder
    from app.retrieval import CrossEncoderReranker
    from eval.hard_negatives import mine_examples
    from eval.qasper import index_papers, load_split
    from eval.run_qasper import (
        candidates_ranker,
        evaluate,
        hit_at_1,
        hybrid_ranker,
        paired_bootstrap,
        reciprocal_rank,
        summary_row,
    )
    from eval.train_embedder import train_embedder
    from eval.train_reranker import train_listwise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    log(f"environment: {environment}")

    splits = {
        name: load_split(name, download_qasper(SCRATCH / "qasper")) for name in ("train", "dev", "test")
    }
    base_embedder = Embedder(BASE_EMBEDDER, device=device)
    base_reranker = CrossEncoderReranker(BASE_RERANKER, device=device)
    base_index = {name: index_papers(papers, base_embedder) for name, papers in splits.items()}

    def end_to_end_mrr(embedder, indexed, reranker) -> float:
        return evaluate(indexed, hybrid_ranker(embedder, reranker))["MRR@10"]

    baseline_dev = end_to_end_mrr(base_embedder, base_index["dev"], base_reranker)
    log(f"dev MRR@10, off-the-shelf embedder and reranker: {baseline_dev:.4f}")

    def embedder_dev_score(checkpoint: Path) -> float:
        embedder = Embedder(str(checkpoint), device=device)
        return end_to_end_mrr(embedder, index_papers(splits["dev"], embedder), base_reranker)

    def reranker_dev_score(checkpoint: Path) -> float:
        reranker = CrossEncoderReranker(str(checkpoint), device=device)
        return end_to_end_mrr(base_embedder, base_index["dev"], reranker)

    runs = {"embedder": [], "reranker": []}
    for variant, denoise in VARIANTS.items():
        teacher = base_reranker if denoise else None
        examples = mine_examples(base_index["train"], base_embedder, teacher=teacher, max_negatives=7)
        negatives = sum(len(example.negatives) for example in examples)
        log(f"{variant}: {len(examples)} training questions, {negatives} hard negatives")
        for component, trainer, scorer, epochs in (
            ("embedder", train_embedder, embedder_dev_score, 3),
            ("reranker", train_listwise, reranker_dev_score, 5),
        ):
            output_dir = SCRATCH / f"{component}-{variant.replace(' ', '-')}"
            base_model = BASE_EMBEDDER if component == "embedder" else BASE_RERANKER
            history = trainer(examples, base_model, output_dir, scorer, epochs=epochs, device=device)
            log(f"{component}, {variant}: {history}")
            runs[component].append(
                {
                    "variant": variant,
                    "questions": len(examples),
                    "hard_negatives": negatives,
                    "history": history,
                    "best": max(history, key=lambda row: row["dev_score"]),
                    "path": output_dir / "best",
                }
            )

    best = {
        component: max(rows, key=lambda run: run["best"]["dev_score"]) for component, rows in runs.items()
    }
    shutil.copytree(best["embedder"]["path"], WORK / "qasper-embedder", dirs_exist_ok=True)
    shutil.copytree(best["reranker"]["path"], WORK / "qasper-reranker-v2", dirs_exist_ok=True)
    tuned_embedder = Embedder(str(WORK / "qasper-embedder"), device=device)
    tuned_reranker = CrossEncoderReranker(str(WORK / "qasper-reranker-v2"), device=device)

    embedders = {"off-the-shelf": base_embedder, "fine-tuned": tuned_embedder}
    rerankers = {"off-the-shelf": base_reranker, "fine-tuned": tuned_reranker}

    def combinations(split: str) -> dict:
        indexes = {
            "off-the-shelf": base_index[split],
            "fine-tuned": index_papers(splits[split], tuned_embedder),
        }
        results = {}
        for e_name, embedder in embedders.items():
            results[f"{e_name} embedder: candidate recall (hybrid top 20)"] = evaluate(
                indexes[e_name], candidates_ranker(embedder), ks=(20,)
            )
            for r_name, reranker in rerankers.items():
                results[f"{e_name} embedder + {r_name} reranker"] = evaluate(
                    indexes[e_name], hybrid_ranker(embedder, reranker)
                )
        return results

    dev = combinations("dev")
    systems = [name for name in dev if "reranker" in name and "candidate" not in name]
    selected = max(systems, key=lambda name: dev[name]["MRR@10"])
    baseline = "off-the-shelf embedder + off-the-shelf reranker"
    log("dev: " + json.dumps([summary_row(k, v) for k, v in dev.items()]))
    log(f"selected on dev: {selected}")

    # The test split is used once, after every choice above was made on dev.
    test = combinations("test")
    log("test: " + json.dumps([summary_row(k, v) for k, v in test.items()]))
    hit_at_4 = lambda rank: float(rank is not None and rank <= 4)  # noqa: E731
    hit_at_20 = lambda rank: float(rank is not None and rank <= 20)  # noqa: E731
    recall = "embedder: candidate recall (hybrid top 20)"
    significance = {
        "selected system vs off-the-shelf": {
            metric: paired_bootstrap(test[baseline]["ranks"], test[selected]["ranks"], fn)
            for metric, fn in (("MRR@10", reciprocal_rank), ("Hit@1", hit_at_1), ("Hit@4", hit_at_4))
        },
        "fine-tuned vs off-the-shelf embedder, candidate recall@20": paired_bootstrap(
            test[f"off-the-shelf {recall}"]["ranks"], test[f"fine-tuned {recall}"]["ranks"], hit_at_20
        ),
    }
    log(f"significance on test: {significance}")

    results = {
        "environment": environment,
        "baseline_dev_mrr": baseline_dev,
        "training": {
            component: [{k: v for k, v in run.items() if k != "path"} for run in rows]
            for component, rows in runs.items()
        },
        "selected_on_dev": selected,
        "dev": dev,
        "test": test,
        "significance_on_test": significance,
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    (WORK / "results_round2.json").write_text(json.dumps(results, indent=1))
    log(f"done in {results['runtime_minutes']} min")


if __name__ == "__main__":
    main()
