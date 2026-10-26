from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from api import main as api_main


@pytest.fixture
def client():
    return TestClient(api_main.app)


@pytest.fixture
def mock_assistant(monkeypatch):
    assistant = Mock()
    assistant.store.list_documents.return_value = [
        {"filename": "README.md", "chunk_count": 3}
    ]
    assistant.store.count_chunks.return_value = 3
    assistant.ingest.return_value = {
        "filename": "sample.txt",
        "chunks_created": 1,
        "status": "indexed",
    }
    assistant.ask.return_value = {
        "answer": "The retention period is 90 days.",
        "sources": [{"filename": "sample.txt"}],
    }

    monkeypatch.setattr(
        api_main, "get_assistant", lambda: assistant
    )
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(max_upload_mb=15),
    )
    return assistant


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "enterprise-knowledge-assistant",
    }


def test_list_documents(client, mock_assistant):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 3
    assert response.json()["documents"][0]["filename"] == "README.md"


def test_upload_document(client, mock_assistant):
    response = client.post(
        "/documents",
        files={"file": ("sample.txt", b"Sample document", "text/plain")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "indexed"
    mock_assistant.ingest.assert_called_once_with(
        "sample.txt", b"Sample document"
    )


def test_upload_rejects_unsupported_file(client, mock_assistant):
    mock_assistant.ingest.side_effect = ValueError(
        "Unsupported file type: .exe"
    )

    response = client.post(
        "/documents",
        files={"file": ("malware.exe", b"test", "application/octet-stream")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_rejects_oversized_file(client, mock_assistant, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(max_upload_mb=1),
    )
    content = b"x" * (1024 * 1024 + 1)

    response = client.post(
        "/documents",
        files={"file": ("large.txt", content, "text/plain")},
    )

    assert response.status_code == 413
    mock_assistant.ingest.assert_not_called()


def test_upload_requires_file(client):
    response = client.post("/documents")

    assert response.status_code == 422


def test_ask_returns_answer_and_sources(client, mock_assistant):
    response = client.post(
        "/ask",
        json={"question": "What is the retention period?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "The retention period is 90 days."
    assert response.json()["sources"][0]["filename"] == "sample.txt"
    mock_assistant.ask.assert_called_once_with(
        "What is the retention period?", None
    )

def test_ask_returns_abstention_response(client, mock_assistant):
    mock_assistant.ask.return_value = {
        "answer": (
            "The provided document context does not contain information "
            "about the company's annual revenue."
        ),
        "sources": [{"filename": "README.md"}],
    }

    response = client.post(
        "/ask",
        json={"question": "What is the company's annual revenue?"},
    )

    assert response.status_code == 200

    body = response.json()
    assert "does not contain information" in body["answer"].lower()
    assert "annual revenue" in body["answer"].lower()
    assert body["sources"][0]["filename"] == "README.md"

    mock_assistant.ask.assert_called_once_with(
        "What is the company's annual revenue?", None
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "A" * 4001},
        {"question": "Valid question", "top_k": 0},
        {"question": "Valid question", "top_k": 21},
    ],
)
def test_ask_rejects_invalid_request(client, payload):
    response = client.post("/ask", json=payload)

    assert response.status_code == 422


def test_ask_handles_service_unavailable(client, mock_assistant):
    mock_assistant.ask.side_effect = RuntimeError("Model unavailable")

    response = client.post(
        "/ask",
        json={"question": "What is the retention period?"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model unavailable"


def test_upload_accepts_file_at_size_limit(client, mock_assistant, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "get_settings",
        lambda: SimpleNamespace(max_upload_mb=1),
    )
    content = b"x" * (1024 * 1024)

    response = client.post(
        "/documents",
        files={"file": ("boundary.txt", content, "text/plain")},
    )

    assert response.status_code == 200
    mock_assistant.ingest.assert_called_once_with("boundary.txt", content)

def test_upload_rejects_malformed_document(client, mock_assistant):
    mock_assistant.ingest.side_effect = ValueError("Invalid PDF document.")

    response = client.post(
        "/documents",
        files={"file": ("broken.pdf", b"invalid pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF document."

def test_upload_handles_unexpected_error(client, mock_assistant):
    mock_assistant.ingest.side_effect = Exception("Sensitive internal details")

    response = client.post(
        "/documents",
        files={"file": ("sample.txt", b"content", "text/plain")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Document ingestion failed. Check the server logs."
    )
    assert "Sensitive internal details" not in response.text

def test_ask_handles_unexpected_error(client, mock_assistant):
    mock_assistant.ask.side_effect = Exception("Sensitive internal details")

    response = client.post(
        "/ask",
        json={"question": "What is the retention period?"},
    )

    assert response.status_code == 500
    assert "Sensitive internal details" not in response.text

def test_ask_rejects_whitespace_question(client):
    response = client.post(
        "/ask",
        json={"question": "   \t\n   "},
    )

    assert response.status_code == 400

def test_ask_rejects_tabs_and_newlines(client):
    response = client.post(
        "/ask",
        json={"question": "\t\n   \t"},
    )

    assert response.status_code == 400


