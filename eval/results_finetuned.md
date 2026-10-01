# Retrieval evaluation with the QASPER fine-tuned models

- **Corpus:** `melodi_iclr2025.pdf` (23 pages), parsed with LiteParse (OCR on)
- **Questions:** 40 hand-written questions in `golden_set.jsonl`, each labelled with verbatim evidence phrases that are checked against the parsed text
- **Relevance:** a retrieved chunk is relevant if it contains an evidence phrase. Hit@k is the share of questions with a relevant chunk in the top k; MRR@10 is the mean reciprocal rank of the first relevant chunk
- **Models:** `eval/.cache/kaggle-round2/qasper-embedder` embeddings, `eval/.cache/kaggle-round2/qasper-reranker-v2` reranker, 20 candidates per retriever before fusion
- **Run:** 2026-10-01, CPU only (11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz). Latency is the median per question and excludes the LLM call
- **Truncated:** share of chunks longer than the embedder's 254-token input; their tails are silently ignored when embedded

## Retrieval ablation

The original system passed its top 3 chunks to the LLM; the upgraded one passes its top 4.

| Configuration | Chunks | Mean tokens | Truncated | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original: 1,000-char windows, dense only | 119 | 228 | 23% | 37.5% | 60.0% | 70.0% | 0.525 | 8 ms |
| Sentence chunks (200 tok), dense only | 140 | 170 | 0% | 47.5% | 72.5% | 77.5% | 0.615 | 8 ms |
| + BM25 hybrid (RRF) | 140 | 170 | 0% | 67.5% | 85.0% | 92.5% | 0.776 | 9 ms |
| + cross-encoder rerank (default) | 140 | 170 | 0% | 80.0% | 92.5% | 95.0% | 0.859 | 504 ms |

## Chunking ablation (hybrid retrieval + reranking)

| Configuration | Chunks | Mean tokens | Truncated | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Median latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000-char windows (original) | 119 | 228 | 23% | 70.0% | 85.0% | 92.5% | 0.783 | 767 ms |
| Sentence chunks, 128 tok / 24 overlap | 224 | 103 | 0% | 72.5% | 90.0% | 92.5% | 0.814 | 336 ms |
| Sentence chunks, 200 tok / 40 overlap (default) | 140 | 170 | 0% | 80.0% | 92.5% | 95.0% | 0.859 | 504 ms |
| Sentence chunks, 254 tok / 48 overlap | 113 | 211 | 0% | 80.0% | 92.5% | 95.0% | 0.859 | 685 ms |

In every configuration each question's evidence lies wholly inside at least one chunk, so every question is answerable in principle.

Questions the default configuration does not retrieve in the top 5: q01, q03.

Reproduce with `python -m eval.run_eval`.
