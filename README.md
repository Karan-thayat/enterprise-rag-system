# Enterprise RAG System

[![CI](https://github.com/Karan-thayat/enterprise-rag-system/actions/workflows/ci.yml/badge.svg)](https://github.com/Karan-thayat/enterprise-rag-system/actions/workflows/ci.yml)

Ask questions about your PDFs and get answers that cite the exact pages they came from.

Documents are parsed, chunked, embedded and searched **locally on CPU**. Hybrid retrieval (BM25 + dense vectors, fused with Reciprocal Rank Fusion) and a cross-encoder reranker pick the passages; only those few passages, the question and recent chat turns are sent to an LLM (OpenAI's open-weight GPT-OSS 20B, served by Groq) to write the answer.

![Asking a question about the sample paper: the answer cites pages, and the sources panel shows the passages](docs/demo.gif)

## Results

**On a public benchmark.** [QASPER](https://huggingface.co/datasets/allenai/qasper) has 1,451 test questions about 416 NLP papers, none of which appear in its train or dev splits. Each question is answered against its own paper with the application's retrieval code, and a passage counts as correct when it comes from a paragraph the annotators marked as evidence ([full report](eval/qasper_results.md)). On the 1,351 test questions with marked evidence:

| Retrieval pipeline | Hit@1 | Hit@4 | MRR@10 |
|---|---:|---:|---:|
| Dense search only | 27.5% | 58.6% | 0.431 |
| Hybrid: BM25 + dense, fused with RRF | 29.2% | 60.3% | 0.448 |
| + off-the-shelf cross-encoder reranker (app default) | 39.9% | 70.3% | 0.544 |
| + embedder and reranker fine-tuned on QASPER's train split | **49.4%** | **81.2%** | **0.642** |

Hit@4 is what the LLM sees. Fine-tuning ran on a Kaggle T4 with every choice made on the dev split; its MRR@10 gain on test is +0.098 (95% CI +0.082 to +0.114). Nearly all of it comes from the reranker, and only after *denoising* its hard negatives. Trained on the raw hard negatives (the top-ranked paragraphs no annotator marked), the reranker scored below the off-the-shelf one on dev (MRR@10 0.487 vs 0.496), likely because annotators don't mark every supporting paragraph; dropping the negatives that the off-the-shelf reranker ranked above all the marked evidence raised it to 0.599.

Answers to 100 random test questions, scored with QASPER's official metrics (the scorer reproduces the official evaluator's numbers exactly): Answer F1 0.247, Evidence F1 0.522. Part of the gap is phrasing, since the official F1 penalises full-sentence answers against short references: on average 56.5% of a reference answer's words appear in the reply, and 7 of 10 yes/no questions are answered correctly. The rest is over-caution: the model refused 21 of 83 answerable questions, 16 of them with passages from the evidence paragraphs in its prompt.

**On the sample paper.** A hand-labelled set of 40 questions about the 23-page MELODI paper ([details](eval/results.md)):

| Retrieval pipeline | Hit@1 | Hit@3 | Hit@5 | MRR@10 |
|---|---:|---:|---:|---:|
| Original: 1,000-character windows, dense search | 27.5% | 52.5% | 57.5% | 0.423 |
| Current: sentence chunks, BM25 + dense (RRF), cross-encoder rerank | **77.5%** | **87.5%** | **95.0%** | **0.837** |

The original system sent its top 3 chunks to the LLM, which contained the answer for only 52.5% of questions. Most of the gain comes from hybrid search and reranking; reranking costs about 0.5 s per question on a laptop CPU. Chunking matters less: under the full pipeline, sentence chunks beat the original windows on Hit@1, Hit@5 and MRR but not on Hit@3, and their main benefits are page-level citations and embeddings that no longer lose the end of long chunks. With 40 questions, one question is worth 2.5 points, so small differences are within noise.

## Example

A real exchange with the sample paper indexed (live run, 30 Sep 2026):

> **Q:** How does MELODI's long-term memory differ from the Memorizing Transformer's?
>
> **A:** MELODI’s long‑term memory differs from the Memorizing Transformer’s in that MELODI stores compressed key‑value (KV) pairs rather than the raw KV pairs of the context tokens. This compression substantially reduces the size of the long‑term memory—e.g., compressing a 512‑token context window into 64 long‑term tokens yields an 8‑fold reduction in memory size [2].
>
> **[2]** melodi_iclr2025.pdf, page 6: “…a key distinction lies in the fact that MELODI stores compressed KV pairs, rather than the KV pairs of context tokens directly, as in MT. This modification substantially reduces the size of the long-term memory…”

Asked who won the 2022 FIFA World Cup, the same system replies “I couldn't find the answer in the uploaded documents.”

## Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingestion: POST /upload"]
        direction TB
        U["PDF bytes"] --> V["Validate: extension, %PDF- header, size limit"]
        V --> H["doc_id = SHA-256 of the bytes; skip if already indexed"]
        H --> P["LiteParse: layout-aware text per page, Tesseract OCR"]
        P --> C["Sentence chunker: at most 200 MiniLM tokens, up to 40 tokens of overlap, page kept"]
        C --> E["all-MiniLM-L6-v2 embeddings (384-d, normalised)"]
    end

    E --> DB[("Chroma: cosine HNSW, ids doc_id:n, filename + page metadata")]
    DB -- "rebuilt after every upload or delete" --> BM[("BM25 index, in memory")]

    subgraph query["Question answering: POST /ask"]
        direction TB
        Q["Question + recent chat history"] --> RW["Follow-up? Rewrite it as a standalone query (LLM)"]
        RW --> D["Dense search: top 20"]
        RW --> K["BM25 search: top 20"]
        D --> F["Reciprocal Rank Fusion"]
        K --> F
        F --> X["Cross-encoder rerank: ms-marco-MiniLM-L-6-v2, keep top 4"]
        X --> G["Prompt: numbered sources with file and page, answer only from them"]
        G --> L["GPT-OSS 20B on Groq, low reasoning effort"]
        L --> A["Answer with [n] citations, sources, per-stage timings"]
    end

    DB --> D
    BM --> K
```

Parsing, embedding, search and reranking run on your machine. The Groq API only ever sees the question, the recent conversation and the four selected passages.

## Features

- **Layout-preserving parsing.** [LiteParse](https://github.com/run-llama/liteparse) places text by its coordinates on the page, so table rows and columns stay aligned in the extracted text, and OCR recovers text inside images.
- **Token-aware chunking.** Chunks are packed from whole sentences up to the embedding model's real limit, and neighbouring chunks share up to 40 tokens of whole sentences. The original fixed 1,000-character windows exceeded MiniLM's 256-token input for 23% of chunks, so their endings were silently dropped from the embedding.
- **Hybrid retrieval.** BM25 catches exact terms (model names, numbers, acronyms) that dense embeddings blur; RRF merges both rankings without score calibration.
- **Reranking.** A cross-encoder reads each question-passage pair jointly to order the 20 fused candidates.
- **Cited, grounded answers.** Sources are numbered `[1]`, `[2]`, … with file name and page, and the prompt instructs the model to answer only from them, cite them, or say the answer isn't in the documents. It also marks retrieved text as untrusted, a basic defence against instructions hidden inside documents.
- **Conversational follow-ups.** "What about its limitations?" is rewritten into a standalone query before retrieval, so the search looks for what the follow-up actually refers to.
- **Document management.** Uploads are de-duplicated by content hash; documents can be listed and deleted, and chunk IDs are namespaced per document so one upload can never overwrite another.
- **Fine-tuned retrieval models (optional).** The jobs in `kaggle/` fine-tune the embedder (contrastive loss with hard negatives) and the reranker (listwise loss with teacher-denoised hard negatives) on QASPER, on a free Kaggle GPU. Both are on the Hugging Face Hub with model cards: [gouneji/all-MiniLM-L6-v2-qasper](https://huggingface.co/gouneji/all-MiniLM-L6-v2-qasper) and [gouneji/ms-marco-MiniLM-L6-v2-qasper](https://huggingface.co/gouneji/ms-marco-MiniLM-L6-v2-qasper). Set `RAG_EMBEDDING_MODEL` and `RAG_RERANKER_MODEL` to those names to use them, and re-upload documents after changing the embedder, because stored vectors come from the old model. On the sample paper, which is not part of QASPER, they also score slightly higher (Hit@3 92.5% vs 87.5%, [details](eval/results_finetuned.md)).
- **Responsive API.** Parsing, embedding and LLM calls run in FastAPI's thread pool, so a long upload doesn't stall other requests: a request sent during a 14-second upload now answers in 0.01 s, where the original `async` endpoints made it wait 12 s.

## Quickstart

Requires Python 3.12, Node.js 18+ (for the LiteParse CLI) and a free [Groq API key](https://console.groq.com). Tested on Linux; the pinned packages also publish wheels for macOS on Apple Silicon and for Windows x64, but those platforms are untested.

```bash
git clone https://github.com/Karan-thayat/enterprise-rag-system.git
cd enterprise-rag-system
python3.12 -m venv .venv             # Windows: py -3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # CPU-only PyTorch, no CUDA download
npm install -g @llamaindex/liteparse@2.15.0
cp .env.example .env                 # then set GROQ_API_KEY
```

No Python 3.12 on your system? With [uv](https://docs.astral.sh/uv/), run `uv venv --python 3.12 .venv` (uv downloads Python 3.12) and install with `uv pip install --index-strategy unsafe-best-match -r requirements.txt`; the flag lets uv combine PyPI with the CPU-only PyTorch index, as pip does.

Start the API and the UI in two terminals:

```bash
uvicorn app.main:app                 # http://127.0.0.1:8000, interactive docs at /docs
streamlit run frontend.py            # opens the chat UI in your browser
```

The first start downloads the embedding and reranking models (about 180 MB) from Hugging Face. Without `GROQ_API_KEY` the API still ingests and searches; only `/ask` returns 503.

Or run both with Docker, which bakes the models into the image and reads `GROQ_API_KEY` from `.env`:

```bash
docker compose up --build            # API on :8000, UI on http://localhost:8501
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Index a PDF (multipart field `file`). Returns the document's ID, pages and chunk count. |
| `POST` | `/ask` | `{"question": "...", "history": [{"role": "user", "content": "..."}]}` → answer, standalone query, numbered sources, timings |
| `POST` | `/search` | `{"query": "...", "top_k": 5}` → ranked passages, no LLM call |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{doc_id}` | Remove a document and its chunks |
| `GET` | `/health` | Document and chunk counts, and whether the LLM is configured |

## Configuration

Settings are read from environment variables or `.env` (see [`.env.example`](.env.example)). The most useful ones:

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | none | Required for `/ask` |
| `RAG_LLM_MODEL` | `openai/gpt-oss-20b` | Groq chat model. Groq retired `llama-3.1-8b-instant` from its free and developer tiers on 16 Aug 2026 and recommends this one instead |
| `RAG_LLM_REASONING_EFFORT` | `low` | Reasoning effort for reasoning models; leave empty for models without reasoning |
| `RAG_TOP_K` | `4` | Passages given to the LLM |
| `RAG_CHUNK_MAX_TOKENS` / `RAG_CHUNK_OVERLAP_TOKENS` | `200` / `40` | Chunk size and overlap, in embedding-model tokens |
| `RAG_USE_HYBRID` / `RAG_USE_RERANKER` | `true` / `true` | Toggle BM25 fusion and reranking |
| `RAG_OCR_ENABLED` | `true` | OCR images while parsing; turning it off makes born-digital PDFs parse much faster |
| `RAG_MAX_UPLOAD_MB` | `25` | Upload size limit; the UI's own limit is `maxUploadSize` in `.streamlit/config.toml` |
| `RAG_API_URL` | `http://127.0.0.1:8000` | Where the Streamlit UI finds the API |

## Evaluation

`python -m eval.run_eval` rebuilds the index for each configuration and scores it against [`eval/golden_set.jsonl`](eval/golden_set.jsonl): 40 questions about the [MELODI paper](eval/data/melodi_iclr2025.pdf) (ICLR 2025), each labelled with verbatim evidence phrases and their page. The script first checks every phrase against the parsed PDF, so a labelling mistake stops the run instead of skewing the numbers. It writes [`eval/results.md`](eval/results.md), which also contains a chunk-size ablation.

For QASPER, `eval/qasper.py` loads the official release and runs the retrieval code per paper, `eval/run_qasper.py` and the scripts in `kaggle/` run the benchmark and the fine-tuning on a Kaggle GPU, `eval/qasper_answers.py` generates and scores answers locally with Groq, and `eval/qasper_report.py` writes [`eval/qasper_results.md`](eval/qasper_results.md) from the raw result files. The exact commands are at the end of that report.

For answer quality beyond QASPER's reference-matching metrics, [`eval/faithfulness.py`](eval/faithfulness.py) is an LLM judge (`gpt-oss-120b`) that splits an answer into claims and checks each one against the passages the model saw. Before using it, it was checked against human hallucination labels from [RAGTruth](https://github.com/ParticleMedia/RAGTruth): on 80 question-answering responses, half of them hallucinated, it agreed with the annotators on 80% (precision 0.80, recall 0.80). [`eval/prompts.py`](eval/prompts.py) holds the answer prompts being compared on QASPER's dev split to reduce over-refusal, and [`eval/answer_report.py`](eval/answer_report.py) turns the runs into a report; that comparison is still running on Groq's free tier.

The two questions the current pipeline still misses in the top 5 are recall failures: the answering passage ranks 51st or lower in both the dense and BM25 lists, so it never reaches the reranker. "What does the name MELODI stand for?" shares no words with the paper's "short for …" phrasing, and "Which research lab are the authors from?" requires knowing that Google DeepMind is a research lab.

## Tests

```bash
pip install -r requirements-dev.txt
pytest                    # 101 tests, offline: no models, parser CLI or API key needed
ruff check . && ruff format --check .
```

The tests swap in a hashing embedder, a fake parser and a fake Groq client. They cover chunking invariants, hybrid retrieval and RRF, citation prompts, follow-up rewriting, every API error path, the Streamlit UI (run headlessly with `AppTest`), the QASPER loading, metrics, hard-negative mining and answer scoring, and the faithfulness judge, its report and the model cards. GitHub Actions runs lint and tests on pushes to `main` and on pull requests, and builds the Docker image, starts it and indexes the sample PDF inside it.

## Project structure

```
app/
  main.py             FastAPI routes and error mapping
  pipeline.py         ingestion and question-answering orchestration
  document_parser.py  LiteParse wrapper (per-page text)
  chunking.py         sentence-aware, token-limited chunker
  embeddings.py       sentence-transformers wrapper
  vector_db.py        Chroma store with per-document IDs and metadata
  retrieval.py        BM25, Reciprocal Rank Fusion, cross-encoder reranker
  llm_generator.py    grounded prompt, citations, follow-up rewriting
  schemas.py          request and response models
  config.py           settings from environment variables
frontend.py           Streamlit chat UI
eval/                 golden question set, QASPER benchmark, training, scoring and faithfulness scripts, model cards, results
kaggle/               GPU jobs for the QASPER benchmark and fine-tuning, and their launcher
docs/                 demo recording
tests/                offline test suite
```

## Limitations and next steps

- **Single tenant, no authentication.** Every user shares one index; per-user collections and auth would come first for real deployment.
- **One process.** Chroma runs embedded and the BM25 index lives in memory, rebuilt after each upload, which is fine for thousands of chunks but not for horizontal scaling. A Chroma server or pgvector plus a search engine would replace both.
- **Answer quality is only partly measured.** The faithfulness judge agrees with human hallucination labels on 80% of RAGTruth responses, but its scores for this application's answers, and a fix for over-refusal (the model declines about a quarter of answerable QASPER questions), are still being measured.
- **Fine-tuned models are trained on NLP papers.** They help on QASPER and don't hurt on the sample paper, but other document types, such as contracts or manuals, are untested.
- **Recall on paraphrased questions.** Fine-tuning raised QASPER candidate recall@20 from 93.3% to 95.8%, but the two sample-paper misses remain; query expansion (e.g. HyDE) would target them, and running page headers could be stripped before indexing.
- **Side-by-side layouts.** Because LiteParse preserves the page's spatial layout, two text columns or a caption beside a paragraph come out interleaved line by line, which can split sentences across chunks.
- **No streaming.** Answers arrive in one piece.
