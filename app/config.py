from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be set with an environment variable prefixed with ``RAG_``
    (e.g. ``RAG_TOP_K=5``) or in a ``.env`` file. The Groq key keeps its
    conventional name, ``GROQ_API_KEY``.
    """

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # Storage
    chroma_path: str = "data/chroma"
    collection_name: str = "documents"

    # Parsing
    liteparse_cli: str | None = None  # path to the LiteParse CLI; auto-detected when unset
    ocr_enabled: bool = True
    parse_timeout_s: float = 300.0
    max_upload_mb: int = 25

    # Chunking, measured in embedding-model tokens
    chunk_max_tokens: int = 200
    chunk_overlap_tokens: int = 40

    # Retrieval
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_hybrid: bool = True
    use_reranker: bool = True
    candidates_k: int = 20
    top_k: int = 4

    # Generation
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    # Groq retired llama-3.1-8b-instant from its free and developer tiers on 2026-08-16;
    # gpt-oss-20b is its recommended replacement. Values follow Groq's reasoning-model guidance.
    llm_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.6
    llm_max_tokens: int = 1024
    llm_reasoning_effort: str | None = "low"  # set to an empty value for models without reasoning
    rewrite_follow_ups: bool = True
    max_history_messages: int = 6


@lru_cache
def get_settings() -> Settings:
    return Settings()
