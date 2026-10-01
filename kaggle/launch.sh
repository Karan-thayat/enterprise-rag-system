#!/usr/bin/env bash
# Starts a job script as a private Kaggle GPU kernel, with the project code embedded in
# the pushed script so the job always runs exactly the code that was launched.
# Requires the Kaggle CLI and an API token in ~/.kaggle/.
#
#   kaggle/launch.sh                                                    # job 1: benchmark + reranker
#   kaggle/launch.sh kaggle/round2_job.py enterprise-rag-qasper-round2  # job 2: second fine-tuning round
#   kaggle kernels status <user>/<slug>                                 # check progress
#   kaggle kernels output <user>/<slug> -p eval/.cache/kaggle-output
set -euo pipefail
cd "$(dirname "$0")/.."

job="${1:-kaggle/qasper_job.py}"
slug="${2:-enterprise-rag-qasper-reranker}"
user="${KAGGLE_USER:-$(kaggle config view | awk '/username:/ {print $3}')}"
kernel="$user/$slug"
build="$(mktemp -d)"
trap 'rm -rf "$build"' EXIT

# Only source files the jobs import: never .env, data or caches.
tar czf "$build/code.tar.gz" app/*.py eval/__init__.py eval/qasper.py eval/run_qasper.py \
  eval/train_reranker.py eval/train_embedder.py eval/hard_negatives.py
echo "Embedding:"; tar tzf "$build/code.tar.gz" | sed 's/^/  /'

mkdir "$build/kernel"
{
  printf 'EMBEDDED_CODE = "%s"  # base64 .tar.gz of the project code, added by kaggle/launch.sh\n\n' \
    "$(base64 -w0 "$build/code.tar.gz")"
  cat "$job"
} > "$build/kernel/$(basename "$job")"

cat > "$build/kernel/kernel-metadata.json" <<EOF
{
  "id": "$kernel",
  "title": "$slug",
  "code_file": "$(basename "$job")",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
kaggle kernels push -p "$build/kernel"
echo "Started $kernel. Check with: kaggle kernels status $kernel"
