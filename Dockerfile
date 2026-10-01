# Enterprise RAG System: API and UI image (CPU only).
FROM python:3.12-slim

# LiteParse, the PDF parser, is a Node.js CLI with a prebuilt native binary.
# Debian's archive only keeps the current package versions, so exact apt pins would break builds.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && npm install -g @llamaindex/liteparse@2.15.0 \
    && apt-get purge -y --auto-remove npm \
    && rm -rf /var/lib/apt/lists/* /root/.npm

RUN useradd --create-home --uid 1000 app
WORKDIR /home/app/service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Numeric, so orchestrators such as Kubernetes can verify the container is not root.
USER 1000:1000
# Bake the embedding and reranking models into the image so containers start without downloading them.
RUN python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && mkdir -p /home/app/data

COPY --chown=1000:1000 app ./app
COPY --chown=1000:1000 frontend.py ./
COPY --chown=1000:1000 .streamlit ./.streamlit

ENV RAG_CHROMA_PATH=/home/app/data/chroma \
    PYTHONUNBUFFERED=1
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
