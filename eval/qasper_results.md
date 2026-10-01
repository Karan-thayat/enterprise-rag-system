# QASPER benchmark

[QASPER](https://huggingface.co/datasets/allenai/qasper) (Dasigi et al., NAACL 2021; CC BY 4.0) has questions written by NLP practitioners who had read only each paper's title and abstract, answered by others who marked the supporting paragraphs in the full text. Each question is answered against its own paper, using the application's retrieval code: 200-token sentence chunks, all-MiniLM-L6-v2 embeddings, BM25, Reciprocal Rank Fusion and a cross-encoder reranker.

| Split | Papers | Questions | With evidence paragraphs |
|---|---:|---:|---:|
| train | 888 | 2593 | 2237 |
| dev | 281 | 1005 | 923 |
| test | 416 | 1451 | 1351 |

**Protocol.** A retrieved chunk counts as relevant when its paragraph (or figure/table caption) was marked as evidence by any annotator; only questions with evidence are scored. The reranker was fine-tuned on the train split, every choice was made on dev, and the test split was evaluated once per round, after that round's choices were fixed. Hit@4 is what the application's LLM sees.

## Retrieval on the test split (1351 questions)

| Retrieval pipeline | Hit@1 | Hit@4 | Hit@10 | MRR@10 |
|---|---:|---:|---:|---:|
| Dense only | 27.5% | 58.6% | 79.8% | 0.431 |
| BM25 only | 23.0% | 51.3% | 74.4% | 0.376 |
| Hybrid (RRF) | 29.2% | 60.3% | 82.2% | 0.448 |
| Hybrid + off-the-shelf reranker | 39.9% | 70.3% | 86.7% | 0.544 |
| Hybrid + fine-tuned reranker | 37.7% | 68.9% | 88.0% | 0.529 |

Fine-tuned against off-the-shelf reranking on the same test questions (paired bootstrap, 10,000 resamples):

| Metric | Fine-tuned minus off-the-shelf | 95% CI |
|---|---:|---:|
| MRR@10 | -0.015 | -0.040 to +0.009 |
| Hit@1 | -2.2 pts | -5.4 pts to +1.0 pts |
| Hit@4 | -1.4 pts | -4.4 pts to +1.6 pts |

## Round 1: fine-tuning the reranker

Starting from `cross-encoder/ms-marco-MiniLM-L-6-v2`, trained with binary cross-entropy on 19,531 (question, chunk) pairs from the train split (3,872 positive). Positives are chunks of evidence paragraphs; negatives are the highest-ranked non-evidence chunks from the hybrid retriever, which is exactly what the reranker has to sort in the application. Batch 32, linear warm-up over 10% of steps, fp16 mixed precision on the GPU.

| Learning rate | Epoch | Train loss | Dev MRR@10 |
|---:|---:|---:|---:|
| 1e-05 | 1 | 0.701 | 0.409 |
| 1e-05 | 2 | 0.438 | 0.455 |
| 1e-05 | 3 | 0.419 | 0.461 |
| 3e-05 | 1 | 0.595 | 0.425 |
| 3e-05 | 2 | 0.391 | 0.472 |
| 3e-05 | 3 (selected) | 0.341 | 0.485 |

Run on Kaggle (Tesla T4, torch 2.10.0+cu128, sentence-transformers 5.4.1) in 19.6 minutes, including all evaluations.

## Answer quality (100 random test questions)

Answers from `openai/gpt-oss-20b` on Groq, using the passages retrieved with the off-the-shelf reranker and the application's own prompt, scored with QASPER's official metrics (best match over annotators). Unanswerable questions are included, as in the official evaluation.

| Metric | Score |
|---|---:|
| Answer F1 | 0.247 |
| Evidence F1 (paragraphs the answer cites) | 0.522 |
| Answer F1, extractive questions | 0.276 |
| Answer F1, abstractive questions | 0.167 |
| Answer F1, boolean questions | 0.063 |
| Answer F1, unanswerable questions | 0.667 |
| Answer token recall (supplementary: gold answer tokens present in the reply) | 0.565 |
| Yes/no accuracy (supplementary: reply starts with the agreed answer) | 7/10 |
| Refused when every annotator marked the question unanswerable | 3/4 |
| Refused when every annotator answered it | 21/83 |

Answer F1 is token overlap with short reference answers, so full-sentence answers score lower than extractive systems even when correct; the per-type rows and refusal counts show where the system actually fails.

**Where the refusals come from.** Of 83 answerable questions, 21 were refused. In 5 the evidence was not among the retrieved passages, so refusing was the grounded choice. In 16 a passage from an evidence paragraph was in the prompt (an upper bound, since a long paragraph can span several chunks), which points at an over-cautious prompt or model rather than retrieval. Tuning the prompt for this belongs on the dev split, not test.

## Round 2: fine-tuning the embedding model, and a listwise reranker

Round 1's reranker lost to the off-the-shelf model, so round 2 changed two things suggested by that result: a listwise loss (softmax over each question's evidence and its hard negatives) instead of pointwise BCE, and hard negatives *denoised* by the off-the-shelf reranker, which drops candidates it ranks above every marked evidence chunk, because QASPER annotators do not mark every supporting paragraph. It also fine-tunes the embedding model, since the reranker can only reorder the 20 candidates the first stage finds. Each component was trained with raw and with denoised negatives, and every epoch was scored on dev. Round 2 was designed after round 1's results, including its test scores, were known; every round-2 choice (variant, epoch and the final combination) was made on dev.

Candidate recall@20 (share of questions with evidence among the 20 chunks the reranker sees):

| Embedder | Dev | Test |
|---|---:|---:|
| off-the-shelf embedder | 92.3% | 93.3% |
| fine-tuned embedder | 94.4% | 95.8% |

Fine-tuned minus off-the-shelf embedder, candidate recall@20 on test: +2.4 pts (95% CI +1.5 to +3.5).

End-to-end retrieval (hybrid + reranking) for every embedder and reranker combination:

| System | Dev MRR@10 | Test Hit@1 | Test Hit@4 | Test Hit@10 | Test MRR@10 |
|---|---:|---:|---:|---:|---:|
| off-the-shelf embedder + off-the-shelf reranker | 0.496 | 39.9% | 70.3% | 86.7% | 0.544 |
| off-the-shelf embedder + fine-tuned reranker | 0.599 | 49.5% | 79.6% | 90.7% | 0.637 |
| fine-tuned embedder + off-the-shelf reranker | 0.504 | 40.1% | 70.8% | 88.4% | 0.549 |
| fine-tuned embedder + fine-tuned reranker **(selected on dev)** | 0.610 | 49.4% | 81.2% | 92.9% | 0.642 |

Selected system minus the off-the-shelf pair on test: MRR@10 +0.098 (95% CI +0.082 to +0.114), Hit@1 +9.5 pts (95% CI +7.2 to +11.9), Hit@4 +10.9 pts (95% CI +8.9 to +12.9).

Training runs (every epoch's checkpoint scored on dev):

| Component | Negatives | Epoch | Train loss | Dev MRR@10 |
|---|---|---:|---:|---:|
| embedder | raw hard negatives | 1 | 3.606 | 0.503 |
| embedder | raw hard negatives | 2 | 3.210 | 0.504 |
| embedder | raw hard negatives | 3 | 3.078 | 0.503 |
| embedder | denoised hard negatives | 1 | 3.455 | 0.501 |
| embedder | denoised hard negatives | 2 | 3.080 | 0.502 |
| embedder | denoised hard negatives | 3 | 2.928 | 0.503 |
| reranker | raw hard negatives | 1 | 2.724 | 0.463 |
| reranker | raw hard negatives | 2 | 1.642 | 0.461 |
| reranker | raw hard negatives | 3 | 1.432 | 0.487 |
| reranker | raw hard negatives | 4 | 1.319 | 0.481 |
| reranker | raw hard negatives | 5 | 1.242 | 0.483 |
| reranker | denoised hard negatives | 1 | 1.676 | 0.568 |
| reranker | denoised hard negatives | 2 | 1.244 | 0.583 |
| reranker | denoised hard negatives | 3 | 1.127 | 0.599 |
| reranker | denoised hard negatives | 4 | 1.001 | 0.595 |
| reranker | denoised hard negatives | 5 | 0.925 | 0.598 |

Run on Kaggle (Tesla T4) in 40.8 minutes.

## Reproduce

```bash
kaggle/launch.sh                                   # GPU job 1: benchmark + reranker (Kaggle account needed)
kaggle kernels output <user>/enterprise-rag-qasper-reranker -p eval/.cache/kaggle-output
kaggle/launch.sh kaggle/round2_job.py enterprise-rag-qasper-round2   # GPU job 2: second round
kaggle kernels output <user>/enterprise-rag-qasper-round2 -p eval/.cache/kaggle-round2
python -m eval.qasper_answers --passages eval/.cache/kaggle-output/answer_passages.json
python -m eval.qasper_report --results eval/.cache/kaggle-output/results.json --answers eval/.cache/qasper_answers.jsonl --passages eval/.cache/kaggle-output/answer_passages.json --round2 eval/.cache/kaggle-round2/results_round2.json
```
