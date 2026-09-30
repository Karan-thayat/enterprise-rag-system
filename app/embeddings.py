from sentence_transformers import SentenceTransformer


class Embedder:
    """A sentence-transformers bi-encoder that produces L2-normalised embeddings."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self._model = SentenceTransformer(model_name, device=device)
        # The model silently truncates inputs beyond max_seq_length, and adds
        # [CLS] and [SEP] itself, so chunks must stay within this many tokens.
        self.max_tokens = self._model.max_seq_length - 2

    def count_tokens(self, text: str) -> int:
        # verbose=False: counting a long sentence is fine; the "longer than the maximum" warning is noise.
        return len(self._model.tokenizer(text, add_special_tokens=False, verbose=False)["input_ids"])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, batch_size=32, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
