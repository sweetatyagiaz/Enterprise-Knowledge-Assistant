
from types import SimpleNamespace

import pytest

from ai_core.rag import KnowledgeAssistant


@pytest.fixture
def make_assistant(monkeypatch):
    """Create a KnowledgeAssistant with mocked embeddings, retrieval, and LLM."""

    def _make_assistant(
        question: str,
        retrieved_chunks: list[dict],
        generated_answer: str,
    ):
        captured_messages = []

        class FakeOpenAI:
            def __init__(self, api_key):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=self.create_completion
                    )
                )

            def create_completion(self, **kwargs):
                captured_messages.extend(kwargs["messages"])
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=generated_answer
                            )
                        )
                    ]
                )

        monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

        assistant = object.__new__(KnowledgeAssistant)
        assistant.settings = SimpleNamespace(
            openai_api_key="test-key",
            chat_model="mock-model",
            top_k=5,
        )
        assistant.embeddings = SimpleNamespace(
            embed_query=lambda query: [1.0, 0.0]
        )
        assistant.store = SimpleNamespace(
            search=lambda query_vector, top_k: retrieved_chunks
        )

        result = assistant.ask(question)

        return result, captured_messages

    return _make_assistant


def test_architecture_answer_is_grounded(make_assistant):
    question = "What are the main components of this project's RAG architecture?"

    chunks = [
        {
            "filename": "README.md",
            "text": (
                "The RAG architecture uses a document loader, text chunking, "
                "OpenAI embeddings, a SQLite vector store, similarity search, "
                "retrieved context, an OpenAI chat model, and FastAPI."
            ),
            "score": 0.9,
        }
    ]

    answer = (
        "The main components are document loading, text chunking, "
        "OpenAI embeddings, SQLite vector storage, similarity search, "
        "retrieved context, the chat model, and FastAPI."
    )

    result, messages = make_assistant(question, chunks, answer)

    assert result["answer"] == answer
    assert result["sources"][0]["filename"] == "README.md"

    user_message = next(
        message["content"]
        for message in messages
        if message["role"] == "user"
    )
    assert chunks[0]["text"] in user_message


def test_embedding_model_answer_is_correct(make_assistant):
    question = "Which embedding model does the project use by default?"

    chunks = [
        {
            "filename": "README.md",
            "text": (
                "The default embedding model is text-embedding-3-small "
                "with 1536 dimensions."
            ),
            "score": 0.9,
        }
    ]

    answer = (
        "The project uses text-embedding-3-small by default, "
        "with 1536-dimensional vectors."
    )

    result, _ = make_assistant(question, chunks, answer)

    assert "text-embedding-3-small" in result["answer"]
    assert "1536" in result["answer"]
    assert result["sources"][0]["filename"] == "README.md"


def test_unknown_revenue_answer_abstains(make_assistant):
    question = "What is the company's annual revenue?"

    chunks = [
        {
            "filename": "README.md",
            "text": (
                "This project is an Enterprise Knowledge Assistant "
                "built with FastAPI, OpenAI embeddings, and SQLite."
            ),
            "score": 0.2,
        }
    ]

    answer = (
        "The provided document context does not contain information "
        "about the company's annual revenue."
    )

    result, messages = make_assistant(question, chunks, answer)

    answer_lower = result["answer"].lower()

    assert "does not contain information" in answer_lower
    assert "revenue" in answer_lower

    user_message = next(
        message["content"]
        for message in messages
        if message["role"] == "user"
    )
    assert "annual revenue" in user_message
    assert "FastAPI" in user_message


def test_unknown_profit_answer_abstains(make_assistant):
    question = "What is the company's annual profit?"

    chunks = [
        {
            "filename": "README.md",
            "text": (
                "This project is an Enterprise Knowledge Assistant "
                "built with FastAPI, OpenAI embeddings, and SQLite."
            ),
            "score": 0.2,
        }
    ]

    answer = (
        "The provided document context does not contain information "
        "about the company's annual profit."
    )

    result, messages = make_assistant(question, chunks, answer)

    answer_lower = result["answer"].lower()

    assert "does not contain information" in answer_lower
    assert "profit" in answer_lower

    user_message = next(
        message["content"]
        for message in messages
        if message["role"] == "user"
    )
    assert "annual profit" in user_message
    assert "FastAPI" in user_message


def test_answer_has_source_information(make_assistant):
    question = "Which database stores document chunks?"

    chunks = [
        {
            "filename": "README.md",
            "text": "SQLite stores document chunks and embedding vectors.",
            "score": 0.95,
        }
    ]

    result, _ = make_assistant(
        question,
        chunks,
        "SQLite stores the document chunks and embedding vectors.",
    )

    assert result["sources"]
    assert result["sources"][0]["filename"] == "README.md"
    assert "SQLite" in result["answer"]

def test_answer_claim_is_supported_by_source_text():
    source_text = (
        "The default embedding model is text-embedding-3-small "
        "with 1536 dimensions."
    )
    answer = "The project uses text-embedding-3-small by default."

    assert "text-embedding-3-small" in source_text
    assert "text-embedding-3-small" in answer


def test_unsupported_claim_is_not_present_in_source_text():
    source_text = (
        "The default embedding model is text-embedding-3-small "
        "with 1536 dimensions."
    )
    answer = "The company's annual profit is $10 million."

    assert "annual profit" not in source_text.casefold()
    assert "$10 million" not in source_text


from ai_core.rag import KnowledgeAssistant
from tests.evaluate_grounding import evaluate_claim


def test_grounding_accepts_supported_paraphrase():
    sources = [
        {
            "filename": "README.md",
            "text": (
                "The default embedding model is text-embedding-3-small "
                "with 1536-dimensional vectors."
            ),
        }
    ]

    result = evaluate_claim(
        "The project uses text-embedding-3-small by default.",
        sources,
    )

    assert result["label"] == "SUPPORTED"
    assert result["evidence"]["filename"] == "README.md"


def test_grounding_rejects_unsupported_number():
    sources = [
        {
            "filename": "README.md",
            "text": "The default embedding vectors have 1536 dimensions.",
        }
    ]

    result = evaluate_claim(
        "The default embedding vectors have 999 dimensions.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"
    assert "999" in result["missing_numbers"]


def test_grounding_accepts_configuration_table_value():
    sources = [
        {
            "filename": "README.md",
            "text": "`CHUNK_SIZE` Characters per text `900`",
        }
    ]

    result = evaluate_claim(
        "The configured text chunk size is 900 characters.",
        sources,
    )

    assert result["label"] == "SUPPORTED"
    assert result["evidence"] is not None


def test_grounding_rejects_number_missing_from_best_evidence():
    sources = [
        {
            "filename": "README.md",
            "text": (
                "The chunk size is 900 characters. "
                "The overlap between chunks is 150 characters."
            ),
        }
    ]

    result = evaluate_claim(
        "The chunk size is 150 characters.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"
    assert "150" in result["missing_numbers"]


def test_grounding_does_not_accept_contradictory_number():
    sources = [
        {
            "filename": "README.md",
            "text": "The maximum upload size is 15 MB.",
        }
    ]

    result = evaluate_claim(
        "The maximum upload size is 50 MB.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"
    assert "50" in result["missing_numbers"]

def test_grounding_supports_sqlite_database_paraphrase():
    claim = (
        "The document chunks and embeddings are stored in a local "
        "SQLite database at data/vector_store.sqlite3."
    )
    sources = [
        {
            "filename": "README.md",
            "text": (
                "The project stores document chunks and embedding vectors "
                "in SQLite at data/vector_store.sqlite3."
            ),
        }
    ]

    result = evaluate_claim(claim, sources)

    assert result["label"] == "SUPPORTED"


def test_grounding_supports_document_format_paraphrase():
    claim = (
        "Users can upload PDF, DOCX, TXT, and Markdown (MD) document formats."
    )
    sources = [
        {
            "filename": "README.md",
            "text": (
                "Document upload for PDF, DOCX, TXT, and Markdown files."
            ),
        }
    ]

    result = evaluate_claim(claim, sources)

    assert result["label"] == "SUPPORTED"

def test_grounding_rejects_incorrect_sqlite_database_path():
    claim = (
        "The document chunks and embeddings are stored in SQLite "
        "at data/incorrect_store.sqlite3."
    )
    sources = [
        {
            "filename": "README.md",
            "text": (
                "The project stores document chunks and embedding vectors "
                "in SQLite at data/vector_store.sqlite3."
            ),
        }
    ]

    result = evaluate_claim(claim, sources)

    assert result["label"] != "SUPPORTED"


def test_grounding_rejects_unsupported_upload_format():
    claim = (
        "Users can upload PDF, DOCX, TXT, Markdown, and CSV documents."
    )
    sources = [
        {
            "filename": "README.md",
            "text": (
                "Document upload for PDF, DOCX, TXT, and Markdown files."
            ),
        }
    ]

    result = evaluate_claim(claim, sources)

    assert result["label"] != "SUPPORTED"


def test_grounding_rejects_negated_numeric_fact():
    sources = [
        {
            "filename": "README.md",
            "text": "The maximum upload size is 15 MB.",
        }
    ]

    result = evaluate_claim(
        "The maximum upload size is not 15 MB.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"


def test_grounding_rejects_negated_database_claim():
    sources = [
        {
            "filename": "README.md",
            "text": "The project uses SQLite for document storage.",
        }
    ]

    result = evaluate_claim(
        "The project does not use SQLite for document storage.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"


def test_grounding_accepts_supported_negative_statement():
    sources = [
        {
            "filename": "README.md",
            "text": "The project does not store uploaded files permanently.",
        }
    ]

    result = evaluate_claim(
        "The project does not store uploaded files permanently.",
        sources,
    )

    assert result["label"] == "SUPPORTED"

def test_grounding_rejects_wrong_configuration_attribution():
    sources = [
        {
            "filename": "README.md",
            "text": (
                "CHUNK_SIZE is 900 characters. "
                "CHUNK_OVERLAP is 150 characters."
            ),
        }
    ]

    result = evaluate_claim(
        "CHUNK_SIZE is 150 characters.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"

def test_grounding_rejects_wrong_model_attribution():
    sources = [
        {
            "filename": "README.md",
            "text": (
                "The embedding model is text-embedding-3-small. "
                "The chat model is gpt-4.1-mini."
            ),
        }
    ]

    result = evaluate_claim(
        "The embedding model is gpt-4.1-mini.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"

def test_grounding_rejects_wrong_model_attribution_across_sources():
    sources = [
        {
            "filename": "architecture.md",
            "text": "The embedding model is text-embedding-3-small.",
        },
        {
            "filename": "configuration.md",
            "text": "The chat model is gpt-4.1-mini.",
        },
    ]

    result = evaluate_claim(
        "The embedding model is gpt-4.1-mini.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"


def test_grounding_rejects_wrong_configuration_value_across_sources():
    sources = [
        {
            "filename": "settings.py",
            "text": "CHUNK_SIZE is 900 characters.",
        },
        {
            "filename": "settings.py",
            "text": "CHUNK_OVERLAP is 150 characters.",
        },
    ]

    result = evaluate_claim(
        "CHUNK_SIZE is 150 characters.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"


def test_grounding_rejects_model_dimension_from_unrelated_source():
    sources = [
        {
            "filename": "models.md",
            "text": "The embedding model is text-embedding-3-small.",
        },
        {
            "filename": "architecture.md",
            "text": "The image encoder produces 768-dimensional vectors.",
        },
    ]

    result = evaluate_claim(
        "The text-embedding-3-small model produces 768-dimensional vectors.",
        sources,
    )

    assert result["label"] == "UNSUPPORTED"


def test_grounding_rejects_unsupported_retrieval_accuracy():
    sources = [
        {
            "filename": "README.md",
            "text": (
                "The application uses FastAPI, OpenAI embeddings, "
                "SQLite, and a retrieval-augmented generation pipeline."
            ),
        }
    ]

    result = evaluate_claim(
        "The application achieves 99.9% retrieval accuracy.",
        sources,
    )

    assert result["label"] in {"UNSUPPORTED", "UNCERTAIN"}

def test_empty_retrieval_returns_no_sources(make_assistant):
    result, _ = make_assistant(
        "What is the annual revenue?",
        [],
        "This answer should not be used when retrieval is empty.",
    )

    assert result["sources"] == []
    assert result["answer"] == (
        "I could not find relevant information in the uploaded documents."
    )

def test_empty_retrieval_does_not_call_llm(make_assistant, monkeypatch):
    calls = []

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: calls.append(kwargs)
                )
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result, messages = make_assistant(
        "What is the annual revenue?",
        [],
        "Unused generated answer.",
    )

    assert result["sources"] == []
    assert calls == []
    assert messages == []

@pytest.mark.parametrize(
    ("question", "generated_answer", "financial_terms"),
    [
        (
            "What is the company's annual revenue?",
            "The company's annual revenue is ₹50 crore.",
            ["revenue", "₹50 crore"],
        ),
        (
            "What is the company's annual profit?",
            "The company's annual profit is ₹10 crore.",
            ["profit", "₹10 crore"],
        ),
    ],
)
def test_unknown_financial_answers_are_not_automatically_verified(
    make_assistant, question, generated_answer, financial_terms
):
    chunks = [
        {
            "filename": "README.md",
            "text": (
                "This project is an Enterprise Knowledge Assistant "
                "built with FastAPI, OpenAI embeddings, and SQLite."
            ),
            "score": 0.2,
        }
    ]

    result, _ = make_assistant(question, chunks, generated_answer)

    # The mocked LLM output is returned unchanged by ask().
    # This test documents the gap: answer generation alone does not
    # guarantee that unsupported financial claims are rejected.
    assert result["answer"] == generated_answer
    assert all(term in result["answer"] for term in financial_terms)



