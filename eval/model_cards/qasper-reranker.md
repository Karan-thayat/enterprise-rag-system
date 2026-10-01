---
license: apache-2.0
language: en
library_name: sentence-transformers
pipeline_tag: text-ranking
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
datasets:
- allenai/qasper
tags:
- sentence-transformers
- cross-encoder
- reranker
- retrieval-augmented-generation
- scientific-papers
---

# ms-marco-MiniLM-L6-v2, fine-tuned on QASPER

A cross-encoder that reranks passages of a scientific paper by how well they answer a question. It is [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) fine-tuned on the train split of [QASPER](https://huggingface.co/datasets/allenai/qasper), built for the retrieval stage of [Enterprise RAG System](https://github.com/Karan-thayat/enterprise-rag-system). There it reorders the 20 candidates that hybrid search (BM25 and dense retrieval, fused with Reciprocal Rank Fusion) finds for a question, and the top 4 go to the language model.

## Usage

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("{repo_id}")
scores = model.predict(
    [
        ("How large is the training set?", "The training set contains 12,000 annotated sentences."),
        ("How large is the training set?", "We use a learning rate of 0.001 and a batch size of 32."),
    ]
)
```

In the application, set `RAG_RERANKER_MODEL={repo_id}`.

## Results on the QASPER test split

The test split has 416 papers, none of them in the train or dev splits; its 1,351 questions with evidence paragraphs are scored. Each question is answered against its own paper, split into chunks of at most 200 tokens, and a chunk is relevant when its paragraph was marked as evidence by an annotator.

| Retrieval pipeline | Hit@1 | Hit@4 | MRR@10 |
|---|---:|---:|---:|
| Hybrid search + the base reranker | 39.9% | 70.3% | 0.544 |
| Hybrid search + this reranker | 49.5% | 79.6% | 0.637 |
| Hybrid search with the [fine-tuned embedder]({embedder_url}) + this reranker | 49.4% | 81.2% | 0.642 |

Hit@4 is the share of questions with an evidence chunk among the 4 passages the language model sees. The last row is the system selected on dev; against the base pair, its MRR@10 gain is +0.098 (paired bootstrap 95% CI +0.082 to +0.114).

## Training

- **Data:** QASPER's train split: 888 papers and 2,237 questions with evidence paragraphs.
- **Examples:** for each question, a chunk from an evidence paragraph and up to 7 hard negatives: the highest-ranked non-evidence chunks from the hybrid retriever, which are exactly what the reranker has to sort in the application.
- **Denoising:** QASPER annotators do not mark every paragraph that supports an answer, so some hard negatives are relevant. Negatives that the base reranker scored at least as high as the best evidence chunk were dropped, leaving 14,666.
- **Loss:** listwise softmax cross-entropy. Each question's evidence chunk competes with its hard negatives, 8 questions per batch.
- **Optimisation:** AdamW, learning rate 2e-5, weight decay 0.01, linear warm-up over 10% of steps, gradient clipping at 1.0, fp16 mixed precision, maximum length 320 tokens, 5 epochs on a Tesla T4.
- **Selection:** every epoch was scored on the dev split; epoch 3 had the best end-to-end dev MRR@10, 0.599 against 0.496 for the base model. The test split was evaluated once, after selection.

The same recipe without denoising scored 0.487 on dev, below the base model, as did pointwise binary cross-entropy training (0.485).

## Limitations

- Trained and tested on NLP research papers in English. Together with the fine-tuned embedder it also helped on the project's own 23-page sample paper, which is not in QASPER (Hit@3 92.5% against 87.5% over 40 questions), but that sample is small.
- It scores single passages of up to 320 tokens, so evidence spread across a long paragraph or a table can be missed.
- QASPER's evidence labels are incomplete, so some passages counted as misses do answer the question.

## Citation

QASPER: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt Gardner. *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers.* NAACL 2021. The dataset is licensed under CC BY 4.0.
