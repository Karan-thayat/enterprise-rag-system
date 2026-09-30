import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Annotated

import groq
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from liteparse import CLINotFoundError, ParseError

from app.config import Settings, get_settings
from app.llm_generator import LLMEmptyAnswerError, LLMNotConfiguredError
from app.pipeline import EmptyDocumentError, RAGPipeline, build_pipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    HealthResponse,
    SearchRequest,
    SearchResponse,
    Source,
    UploadResponse,
)
from app.vector_db import RetrievedChunk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


Pipeline = Annotated[RAGPipeline, Depends(get_pipeline)]


def create_app(
    settings: Settings | None = None,
    pipeline_factory: Callable[[Settings], RAGPipeline] = build_pipeline,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info("Loading models and opening the vector store...")
        app.state.pipeline = pipeline_factory(settings)
        yield

    app = FastAPI(
        title="Enterprise RAG System API",
        version="2.0.0",
        description="Upload PDFs and ask questions that are answered with cited sources.",
        lifespan=lifespan,
    )

    # Routes are plain `def` functions on purpose: parsing, embedding and LLM calls
    # block, so FastAPI runs them in its thread pool instead of on the event loop.

    @app.get("/health")
    def health(pipeline: Pipeline) -> HealthResponse:
        return HealthResponse(
            status="ok",
            documents=len(pipeline.list_documents()),
            chunks=pipeline.count_chunks(),
            llm_configured=pipeline.llm_configured,
        )

    @app.post("/upload")
    def upload_pdf(file: Annotated[UploadFile, File()], pipeline: Pipeline) -> UploadResponse:
        filename = os.path.basename((file.filename or "").replace("\\", "/")) or "document.pdf"
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Only PDF files are supported.")

        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"PDFs larger than {settings.max_upload_mb} MB are not accepted."
            )
        if b"%PDF-" not in data[:1024]:
            raise HTTPException(status_code=415, detail="The file is not a valid PDF.")

        try:
            result = pipeline.ingest_pdf(data, filename)
        except EmptyDocumentError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except (ParseError, TimeoutError) as error:
            logger.warning("Could not parse %s: %s", filename, getattr(error, "stderr", None) or error)
            raise HTTPException(status_code=422, detail="The PDF could not be parsed.") from error
        except CLINotFoundError as error:
            logger.error("%s", error)
            raise HTTPException(
                status_code=503, detail="The document parser (LiteParse CLI) is not installed on the server."
            ) from error

        document = result.document
        if result.already_indexed:
            message = f"{document.filename} is already indexed."
        else:
            message = f"Indexed {document.filename}: {document.pages} pages, {document.chunks} chunks."
        return UploadResponse(
            document=DocumentInfo(**asdict(document)),
            already_indexed=result.already_indexed,
            message=message,
        )

    @app.post("/ask")
    def ask(body: AskRequest, pipeline: Pipeline) -> AskResponse:
        history = [message.model_dump() for message in body.history]
        try:
            result = pipeline.answer(body.question, history)
        except LLMNotConfiguredError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except LLMEmptyAnswerError as error:
            logger.warning("%s", error)
            raise HTTPException(
                status_code=502, detail="The LLM returned an empty answer; try again."
            ) from error
        except groq.RateLimitError as error:
            raise HTTPException(
                status_code=429, detail="The LLM provider is rate limiting requests; try again shortly."
            ) from error
        except groq.APIError as error:
            logger.error("LLM provider error: %s", error)
            raise HTTPException(status_code=502, detail="The LLM provider returned an error.") from error
        return AskResponse(
            question=result.question,
            standalone_question=result.standalone_question,
            answer=result.answer,
            sources=_to_sources(result.sources),
            timings_ms=result.timings_ms,
        )

    @app.post("/search")
    def search(body: SearchRequest, pipeline: Pipeline) -> SearchResponse:
        return SearchResponse(query=body.query, results=_to_sources(pipeline.search(body.query, body.top_k)))

    @app.get("/documents")
    def list_documents(pipeline: Pipeline) -> list[DocumentInfo]:
        return [DocumentInfo(**asdict(document)) for document in pipeline.list_documents()]

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, pipeline: Pipeline) -> Response:
        if not pipeline.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found.")
        return Response(status_code=204)

    return app


def _to_sources(chunks: list[RetrievedChunk]) -> list[Source]:
    return [
        Source(
            number=number,
            chunk_id=chunk.id,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            page=chunk.page,
            text=chunk.text,
            score=chunk.score,
        )
        for number, chunk in enumerate(chunks, start=1)
    ]


app = create_app()
