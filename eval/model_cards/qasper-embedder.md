---
license: apache-2.0
language: en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: sentence-transformers/all-MiniLM-L6-v2
datasets:
- allenai/qasper
tags:
- sentence-transformers
- feature-extraction
- retrieval-augmented-generation
- scientific-papers
---

# all-MiniLM-L6-v2, fine-tuned on QASPER

A sentence-embedding model for finding the passages of a scientific paper that answer a question. It is [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) fine-tuned on the train split of [QASPER](https://huggingface.co/datasets/allenai/qasper), built for the dense half of the hybrid search in [Enterprise RAG System](https://github.com/Karan-thayat/enterprise-rag-system). It produces 384-dimensional, unit-length embeddings, like the base model.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{repo_id}")
embeddings = model.encode(
    [
        "How large is the training set?",
        "The training set contains 12,000 annotated sentences.",
    ]
)
```

In the application, set `RAG_EMBEDDING_MODEL={repo_id}` and upload your documents again: stored vectors from another model are not comparable.

## Results on the QASPER test split

The test split has 416 papers, none of them in the train or dev splits; its 1,351 questions with evidence paragraphs are scored. Each question is answered against its own paper, split into chunks of at most 200 tokens, and a chunk is relevant when its paragraph was marked as evidence by an annotator.

The embedder decides which chunks reach the reranker: hybrid search keeps 20 candidates.

| Embedder | Candidate recall@20 | MRR@10 with the base reranker | MRR@10 with the [fine-tuned reranker]({reranker_url}) |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 93.3% | 0.544 | 0.637 |
| this model | 95.8% | 0.549 | 0.642 |

Candidate recall@20 is the share of questions with an evidence chunk among the 20 candidates; the gain is +2.4 points (paired bootstrap 95% CI +1.5 to +3.5). Most of the end-to-end gain comes from the reranker, which reorders those candidates.

## Training

- **Data:** QASPER's train split: 888 papers and 2,237 questions with evidence paragraphs.
- **Examples:** for each question, a chunk from an evidence paragraph and up to 7 hard negatives: the highest-ranked non-evidence chunks from the hybrid retriever (15,659 in all).
- **Loss:** contrastive (multiple-negatives ranking loss) with a scale of 20. In each batch of 64 questions, a question's evidence chunk competes with one of its hard negatives and with every other passage in the batch.
- **Optimisation:** AdamW, learning rate 2e-5, weight decay 0.01, linear warm-up over 10% of steps, gradient clipping at 1.0, fp16 mixed precision, 3 epochs on a Tesla T4.
- **Selection:** every epoch was scored on the dev split by end-to-end MRR@10 with the base reranker; epoch 2 scored best (0.504 against 0.496 for the base model). A variant trained on negatives denoised by the base reranker scored 0.503. The test split was evaluated once, after selection.

## Limitations

- Trained and tested on NLP research papers in English. On the project's own 23-page sample paper, which is not in QASPER, dense-only retrieval improved from MRR@10 0.434 to 0.615 over 40 questions, but that sample is small.
- Inputs longer than 256 word pieces are truncated, as in the base model.
- QASPER's evidence labels are incomplete, so some passages counted as misses do answer the question.

## Citation

QASPER: Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, Matt Gardner. *A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers.* NAACL 2021. The dataset is licensed under CC BY 4.0.
