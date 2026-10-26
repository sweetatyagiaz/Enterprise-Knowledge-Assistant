
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ai_core.rag import KnowledgeAssistant
from ai_core.settings import get_settings

app = FastAPI(
    title="Enterprise Knowledge Assistant",
    version="0.1.0",
    description="RAG application with document ingestion, retrieval, and citations.",
)


@lru_cache
def get_assistant() -> KnowledgeAssistant:
    return KnowledgeAssistant()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
    }


@app.get("/documents")
def list_documents():
    try:
        assistant = get_assistant()
        return {
            "documents": assistant.store.list_documents(),
            "chunk_count": assistant.store.count_chunks(),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not access the document store.",
        ) from exc


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()
    filename = file.filename or "upload.txt"

    content = await file.read(
        settings.max_upload_mb * 1024 * 1024 + 1
    )

    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_mb} MB limit.",
        )

    try:
        return get_assistant().ingest(filename, content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        import logging
        logging.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document ingestion failed. Check the server logs."
        ) from exc


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return get_assistant().ask(request.question, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Question answering failed.",
        ) from exc
