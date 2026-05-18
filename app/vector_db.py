import chromadb
from sentence_transformers import SentenceTransformer

class VectorDBClient:
    def __init__(self):
        # 1. Initialize the embedding model on the CPU (Kept "warm" in memory)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # 2. Connect to the local Chroma database
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="pdf_knowledge_base")

    def upsert_chunks(self, chunks: list[str]) -> int:
        """Converts a list of strings into embeddings and saves them."""
        if not chunks:
            return 0
            
        embeddings = self.embedding_model.encode(chunks).tolist()
        chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks
        )
        return len(chunks)

    def search(self, query: str, n_results: int = 3) -> str:
        """Takes a user question, searches the DB, and returns the combined text."""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # If the database is empty or no results match, return a fallback string
        if not results['documents'] or not results['documents'][0]:
            return "No relevant context found."
            
        # Join the top chunks into a single clean string
        return "\n\n".join(results['documents'][0])