"""Kaggle GPU job: QASPER retrieval benchmark and reranker fine-tuning.

Launched by kaggle/launch.sh, which uploads the project code as a private dataset
and pushes this script as a private GPU kernel. Protocol:
  1. Baselines on the dev split.
  2. Fine-tune the reranker on the train split with a small learning-rate grid;
     pick the checkpoint with the best dev MRR@10.
  3. Evaluate every configuration once on the test split, with paired bootstrap
     confidence intervals for fine-tuned vs. off-the-shelf reranking.
  4. Export the retrieved passages for 100 random test questions; the answers are
     generated and scored locally, so no API key is needed here.

Outputs in /kaggle/working: results.json, answer_passages.json, qasper-reranker/.
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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LEARNING_RATES = [1e-5, 3e-5]
EPOCHS = 3
ANSWER_SAMPLE = 100


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
    from eval.qasper import index_papers, load_split
    from eval.run_qasper import (
        bm25_ranker,
        dense_ranker,
        evaluate,
        export_passages,
        hit_at_1,
        hybrid_ranker,
        paired_bootstrap,
        reciprocal_rank,
        summary_row,
    )
    from eval.train_reranker import mine_pairs, train

    device = "cuda" if torch.cuda.is_available() else "cpu"
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    log(f"environment: {environment}")

    data_dir = download_qasper(SCRATCH / "qasper")
    splits = {name: load_split(name, data_dir) for name in ("train", "dev", "test")}
    stats = {
        name: {
            "papers": len(papers),
            "questions": sum(len(p.questions) for p in papers),
            "questions_with_evidence": sum(1 for p in papers for q in p.questions if q.evidence),
        }
        for name, papers in splits.items()
    }
    log(f"QASPER splits: {stats}")

    embedder = Embedder(EMBEDDING_MODEL, device=device)
    base_reranker = CrossEncoderReranker(BASE_RERANKER, device=device)
    indexed = {name: index_papers(papers, embedder) for name, papers in splits.items()}
    log("indexed all splits")

    def standard_configs(split: str) -> dict:
        return {
            "Dense only": evaluate(indexed[split], dense_ranker(embedder)),
            "BM25 only": evaluate(indexed[split], bm25_ranker()),
            "Hybrid (RRF)": evaluate(indexed[split], hybrid_ranker(embedder)),
            "Hybrid + off-the-shelf reranker": evaluate(
                indexed[split], hybrid_ranker(embedder, base_reranker)
            ),
        }

    dev = standard_configs("dev")
    log("dev baselines: " + json.dumps([summary_row(k, v) for k, v in dev.items()]))

    pairs = mine_pairs(indexed["train"], embedder)
    log(f"mined {len(pairs)} training pairs ({sum(p.label == 1 for p in pairs)} positive)")

    def dev_mrr(checkpoint: Path) -> float:
        reranker = CrossEncoderReranker(str(checkpoint), device=device)
        return evaluate(indexed["dev"], hybrid_ranker(embedder, reranker))["MRR@10"]

    runs = []
    for learning_rate in LEARNING_RATES:
        output_dir = SCRATCH / f"reranker-lr{learning_rate}"
        history = train(
            pairs,
            BASE_RERANKER,
            output_dir,
            dev_mrr,
            learning_rate=learning_rate,
            epochs=EPOCHS,
            device=device,
        )
        best = max(history, key=lambda row: row["dev_score"])
        runs.append(
            {"learning_rate": learning_rate, "history": history, "best": best, "path": output_dir / "best"}
        )
        log(f"lr={learning_rate}: {history}")
    selected = max(runs, key=lambda run: run["best"]["dev_score"])
    shutil.copytree(selected["path"], WORK / "qasper-reranker", dirs_exist_ok=True)
    tuned_reranker = CrossEncoderReranker(str(WORK / "qasper-reranker"), device=device)
    dev["Hybrid + fine-tuned reranker"] = evaluate(indexed["dev"], hybrid_ranker(embedder, tuned_reranker))
    log(f"selected lr={selected['learning_rate']} epoch={selected['best']['epoch']}")

    # The test split is used once, after every choice above was made on dev.
    test = standard_configs("test")
    test["Hybrid + fine-tuned reranker"] = evaluate(indexed["test"], hybrid_ranker(embedder, tuned_reranker))
    base_ranks = test["Hybrid + off-the-shelf reranker"]["ranks"]
    tuned_ranks = test["Hybrid + fine-tuned reranker"]["ranks"]
    significance = {
        "MRR@10": paired_bootstrap(base_ranks, tuned_ranks, reciprocal_rank),
        "Hit@1": paired_bootstrap(base_ranks, tuned_ranks, hit_at_1),
        "Hit@4": paired_bootstrap(base_ranks, tuned_ranks, lambda r: float(r is not None and r <= 4)),
    }
    log("test: " + json.dumps([summary_row(k, v) for k, v in test.items()]))
    log(f"fine-tuned minus off-the-shelf on test: {significance}")

    use_tuned = (
        dev["Hybrid + fine-tuned reranker"]["MRR@10"] > dev["Hybrid + off-the-shelf reranker"]["MRR@10"]
    )
    answer_reranker = tuned_reranker if use_tuned else base_reranker
    passages = export_passages(indexed["test"], hybrid_ranker(embedder, answer_reranker), ANSWER_SAMPLE)
    (WORK / "answer_passages.json").write_text(json.dumps(passages, indent=1))

    results = {
        "environment": environment,
        "splits": stats,
        "training": {
            "pairs": len(pairs),
            "positive_pairs": sum(p.label == 1 for p in pairs),
            "runs": [{k: v for k, v in run.items() if k != "path"} for run in runs],
            "selected": {"learning_rate": selected["learning_rate"], **selected["best"]},
        },
        "dev": dev,
        "test": test,
        "significance_test_tuned_vs_base": significance,
        "answer_scoring_reranker": "fine-tuned" if use_tuned else "off-the-shelf",
        "runtime_minutes": round((time.time() - started) / 60, 1),
    }
    (WORK / "results.json").write_text(json.dumps(results, indent=1))
    log(f"done in {results['runtime_minutes']} min")


if __name__ == "__main__":
    main()
