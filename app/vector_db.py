from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.chunking import Chunk

if TYPE_CHECKING:
    from app.embeddings import Embedder


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    doc_id: str
    filename: str
    page: int
    score: float = 0.0


@dataclass(frozen=True)
class DocumentSummary:
    doc_id: str
    filename: str
    pages: int
    chunks: int


class VectorStore:
    """A persistent Chroma collection of chunk embeddings plus citation metadata."""

    def __init__(self, path: str, collection_name: str, embedder: "Embedder"):
        # Imported here so the lightweight parts of this module (RetrievedChunk) can be
        # used by the evaluation and training scripts without installing Chroma.
        import chromadb

        self.embedder = embedder
        self._client = chromadb.PersistentClient(
            path=path, settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,  # vectors always come from `embedder`
        )

    def count(self) -> int:
        return self._collection.count()

    def add_document(self, doc_id: str, filename: str, num_pages: int, chunks: list[Chunk]) -> None:
        """Embeds and stores a document's chunks.

        Chunk IDs are namespaced by ``doc_id`` (``<doc_id>:<chunk index>``), so a
        new document can never overwrite the chunks of another one.
        """
        ids = [f"{doc_id}:{chunk.index}" for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas: list[dict[str, Any]] = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "num_pages": num_pages,
                "page": chunk.page,
                "chunk_index": chunk.index,
            }
            for chunk in chunks
        ]
        embeddings = self.embedder.embed_documents(texts)
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    def delete_document(self, doc_id: str) -> int:
        """Deletes every chunk of a document and returns how many were removed."""
        ids = self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_document(self, doc_id: str) -> DocumentSummary | None:
        documents = self.list_documents(where={"doc_id": doc_id})
        return documents[0] if documents else None

    def list_documents(self, where: dict[str, Any] | None = None) -> list[DocumentSummary]:
        metadatas = self._collection.get(where=where, include=["metadatas"])["metadatas"]
        chunk_counts = Counter(metadata["doc_id"] for metadata in metadatas)
        first_seen: dict[str, dict[str, Any]] = {}
        for metadata in metadatas:
            first_seen.setdefault(metadata["doc_id"], metadata)
        documents = [
            DocumentSummary(
                doc_id=doc_id,
                filename=metadata["filename"],
                pages=metadata["num_pages"],
                chunks=chunk_counts[doc_id],
            )
            for doc_id, metadata in first_seen.items()
        ]
        return sorted(documents, key=lambda document: (document.filename.lower(), document.doc_id))

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        """Returns up to ``k`` nearest chunks, scored by cosine similarity."""
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        return [
            _to_chunk(chunk_id, text, metadata, score=1.0 - distance)
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def all_chunks(self) -> list[RetrievedChunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            _to_chunk(chunk_id, text, metadata)
            for chunk_id, text, metadata in zip(
                result["ids"], result["documents"], result["metadatas"], strict=True
            )
        ]


def _to_chunk(chunk_id: str, text: str, metadata: dict[str, Any], score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        text=text,
        doc_id=metadata["doc_id"],
        filename=metadata["filename"],
        page=metadata["page"],
        score=score,
    )
