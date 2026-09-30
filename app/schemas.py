from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    # Only conversation turns are accepted; a client cannot inject "system" messages.
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=50)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    number: int = Field(description="Citation number used in the answer, e.g. [1]")
    chunk_id: str
    doc_id: str
    filename: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: list[Source]
    timings_ms: dict[str, float]


class SearchResponse(BaseModel):
    query: str
    results: list[Source]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int


class UploadResponse(BaseModel):
    document: DocumentInfo
    already_indexed: bool
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    chunks: int
    llm_configured: bool
