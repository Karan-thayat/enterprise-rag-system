# Enterprise RAG System 

An end-to-end, production-grade Retrieval-Augmented Generation (RAG) system. This application allows users to upload complex PDF documents and ask questions about them, utilizing local vector mathematics and a cloud-based LLM to generate highly accurate, context-aware answers.

---

## Core Features

- **Interactive Chat Interface:** A sleek, ChatGPT-style web UI built with Streamlit for seamless user interaction.
- **Contextual Memory:** The system remembers previous interactions within a session, allowing for natural conversational follow-up questions.
- **Dynamic Document Upload:** Drag-and-drop PDF ingestion directly from the UI.
- **Instant Answer Generation:** Powered by the ultra-fast Groq API running the `llama-3.1-8b-instant` model.

---

## ⚡ Why This Architecture?

### 1. Spatial Document Parsing (LiteParse) vs. Naive Extraction

#### The Problem
Standard parsers like PyMuPDF or PyPDF2 extract text linearly from top to bottom. Multi-column PDFs, tables, and structured layouts often become scrambled.

#### The Solution
This system uses **LiteParse** to preserve the spatial structure of documents. It evaluates word coordinates and reconstructs text in a layout-aware format, improving retrieval quality and reducing hallucinations.

---

### 2. Hybrid Compute Pipeline (Cost & Privacy Optimized)

#### The Problem
Generating embeddings using cloud APIs for large documents is expensive and raises privacy concerns.

#### The Solution
This system performs:
- Local embedding generation
- Local vector indexing
- Local retrieval

Only the final top-relevant chunks are sent to the cloud LLM for answer generation.

Benefits:
- Lower cost
- Better privacy
- Faster performance

---

### 3. Decoupled Stateless Backend

The backend and frontend are completely separated.

#### Backend
- FastAPI
- Document parsing
- Embedding generation
- Vector database operations
- LLM interaction

#### Frontend
- Streamlit
- Chat UI
- Session memory
- User interaction

This architecture makes the project scalable and maintainable.

---

### 4. Modular Object-Oriented Design

The system is divided into isolated components:

- `DocumentParser`
- `VectorDBClient`
- `LLMClient`

This allows easy replacement of:
- Vector databases
- Embedding models
- LLM providers
- Parsing engines

without changing the rest of the pipeline.

---

## How to Run Locally

### 1. Prerequisites

- Python 3.10+
- Node.js
- Groq API Key

Get your API key here:

https://console.groq.com/

---

### 2. Clone Repository

```bash
git clone <your-repository-url>
cd enterprise-rag-system
```

---

### 3. Install LiteParse Engine

```bash
npm install -g @llamaindex/liteparse
```

---

### 4. Create Virtual Environment

```bash
python3 -m venv .venv
```

---

### 5. Activate Virtual Environment

#### Ubuntu/macOS

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

---

### 6. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## Start the Application

Open **two terminals**.

### Terminal 1 — Start Backend

```bash
uvicorn app.main:app --reload
```

Backend API docs:

```text
http://127.0.0.1:8000/docs
```

---

### Terminal 2 — Start Frontend

```bash
streamlit run frontend.py
```

The Streamlit interface will automatically open in your browser.

---

## Usage

1. Open the Streamlit UI
2. Upload a PDF document
3. Click **Process Document**
4. Wait for processing to complete
5. Ask questions about the uploaded document
