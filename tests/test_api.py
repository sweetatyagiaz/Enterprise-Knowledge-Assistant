from io import BytesIO
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_service_status():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
    }


def test_list_documents_returns_store_metadata(monkeypatch):
    assistant = SimpleNamespace(
        store=SimpleNamespace(
            list_documents=lambda: [
                {"filename": "guide.md", "chunk_count": 2}
            ],
            count_chunks=lambda: 2,
        )
    )
    monkeypatch.setattr("api.main.get_assistant", lambda: assistant)

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {
        "documents": [{"filename": "guide.md", "chunk_count": 2}],
        "chunk_count": 2,
    }


def test_list_documents_returns_503_on_store_error(monkeypatch):
    def fail():
        raise RuntimeError("Database unavailable")

    monkeypatch.setattr("api.main.get_assistant", fail)

    response = client.get("/documents")

    assert response.status_code == 503
    assert response.json()["detail"] == "Could not access the document store."


def test_upload_document_returns_ingestion_result(monkeypatch):
    assistant = SimpleNamespace(
        ingest=lambda filename, content: {
            "filename": filename,
            "chunks_created": 2,
            "status": "indexed",
        }
    )
    monkeypatch.setattr("api.main.get_assistant", lambda: assistant)
    monkeypatch.setattr(
        "api.main.get_settings",
        lambda: SimpleNamespace(max_upload_mb=1),
    )

    response = client.post(
        "/documents",
        files={"file": ("guide.txt", BytesIO(b"Document text"), "text/plain")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "guide.txt",
        "chunks_created": 2,
        "status": "indexed",
    }


def test_upload_document_returns_400_for_invalid_content(monkeypatch):
    def fail_ingest(filename, content):
        raise ValueError("Unsupported document")

    assistant = SimpleNamespace(ingest=fail_ingest)
    monkeypatch.setattr("api.main.get_assistant", lambda: assistant)
    monkeypatch.setattr(
        "api.main.get_settings",
        lambda: SimpleNamespace(max_upload_mb=1),
    )

    response = client.post(
        "/documents",
        files={"file": ("bad.txt", BytesIO(b"invalid"), "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported document"


def test_upload_document_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(
        "api.main.get_settings",
        lambda: SimpleNamespace(max_upload_mb=1),
    )

    response = client.post(
        "/documents",
        files={
            "file": (
                "large.txt",
                BytesIO(b"x" * (1024 * 1024 + 1)),
                "text/plain",
            )
        },
    )

    assert response.status_code == 413


def test_ask_endpoint_returns_answer(monkeypatch):
    assistant = SimpleNamespace(
        ask=lambda question, top_k: {
            "answer": "The capital is Paris.",
            "sources": [{"filename": "geography.txt"}],
        }
    )
    monkeypatch.setattr("api.main.get_assistant", lambda: assistant)

    response = client.post(
        "/ask",
        json={"question": "What is the capital?", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "The capital is Paris."
    assert response.json()["sources"] == [{"filename": "geography.txt"}]


def test_ask_endpoint_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_ask_endpoint_rejects_out_of_range_top_k():
    response = client.post(
        "/ask",
        json={"question": "Find information", "top_k": 21},
    )

    assert response.status_code == 422


def test_ask_endpoint_returns_503_for_runtime_error(monkeypatch):
    def fail(question, top_k):
        raise RuntimeError("Model service unavailable")

    assistant = SimpleNamespace(ask=fail)
    monkeypatch.setattr("api.main.get_assistant", lambda: assistant)

    response = client.post("/ask", json={"question": "Explain RAG"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Model service unavailable"
