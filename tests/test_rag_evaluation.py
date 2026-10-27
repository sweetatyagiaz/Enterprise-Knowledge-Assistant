
from types import SimpleNamespace

import pytest

from ai_core.chunking import chunk_text
from ai_core.ingestion.loaders import load_document
from ai_core.rag import KnowledgeAssistant
from ai_core.vector_store import VectorStore
from ai_core.retrieval.ranking import hybrid_rank


# --------------------------------------------------
# Test 1: Text chunking
# --------------------------------------------------

def test_chunk_text_creates_overlapping_chunks():
    text = "abcdefghij"

    chunks = chunk_text(
        text,
        chunk_size=6,
        chunk_overlap=2,
    )

    assert chunks == ["abcdef", "efghij"]


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text(
            "Sample text",
            chunk_size=10,
            chunk_overlap=10,
        )


def test_chunk_text_handles_empty_input():
    assert chunk_text("   ") == []


# --------------------------------------------------
# Test 2: Document ingestion
# --------------------------------------------------

def test_markdown_loader_extracts_text():
    content = b"# RAG Project\n\nThis project uses embeddings."

    result = load_document("README.md", content)

    assert "RAG Project" in result
    assert "embeddings" in result


def test_loader_rejects_unsupported_file():
    with pytest.raises(ValueError):
        load_document("document.exe", b"test")


def test_loader_rejects_empty_content():
    with pytest.raises(ValueError):
        load_document("README.md", b"")


# --------------------------------------------------
# Test 3: Vector retrieval
# --------------------------------------------------

def test_vector_store_returns_most_similar_chunk(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "test_vectors.sqlite3"

    monkeypatch.setattr(
        "ai_core.vector_store.get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    store = VectorStore()

    store.add_document(
        filename="test.txt",
        chunks=[
            "The capital of France is Paris.",
            "Python is a programming language.",
        ],
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert "Paris" in results[0]["text"]
    assert results[0]["score"] == pytest.approx(1.0)


# --------------------------------------------------
# Test 4: RAG answer generation
# --------------------------------------------------

def test_rag_uses_retrieved_context(monkeypatch):
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
                            content="The capital of France is Paris."
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
        embed_query=lambda question: [1.0, 0.0]
    )

    assistant.store = SimpleNamespace(
        search=lambda query_vector, top_k: [
            {
                "filename": "geography.txt",
                "text": "The capital of France is Paris.",
                "score": 1.0,
            }
        ]
    )

    result = assistant.ask(
        "What is the capital of France?"
    )

    assert "Paris" in result["answer"]
    assert result["sources"][0]["filename"] == "geography.txt"

    # Confirm retrieved context was sent to the model.
    user_message = next(
        message["content"]
        for message in captured_messages
        if message["role"] == "user"
    )

    assert "The capital of France is Paris." in user_message


def test_rag_handles_no_retrieved_results():
    assistant = object.__new__(KnowledgeAssistant)

    assistant.settings = SimpleNamespace(top_k=5)

    assistant.embeddings = SimpleNamespace(
        embed_query=lambda question: [1.0, 0.0]
    )

    assistant.store = SimpleNamespace(
        search=lambda query_vector, top_k: []
    )

    result = assistant.ask("What is the capital of France?")

    assert result["sources"] == []
    assert "could not find relevant information" in (
        result["answer"].lower()
    )


def test_rag_does_not_invent_answer_from_irrelevant_context(monkeypatch):
    """Verify the model is instructed not to invent unsupported facts."""
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
                            content=(
                                "I could not find relevant information "
                                "to answer this question."
                            )
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
        embed_query=lambda question: [1.0, 0.0]
    )
    assistant.store = SimpleNamespace(
        search=lambda query_vector, top_k: [
            {
                "filename": "unrelated.txt",
                "text": "Aurora Analytics has a default retention period of 90 days.",
                "score": 0.3,
            }
        ]
    )

    result = assistant.ask("What is the company's annual revenue?")

    assert result["sources"]
    assert "does not contain information" in result["answer"].lower()
    assert "revenue" in result["answer"].lower()

    # The financial guard abstains before invoking the LLM.
    assert captured_messages == []

def test_markdown_chunking_preserves_heading_with_content():
    text = (
        "# Architecture\n\n"
        "The system uses embeddings and a vector store.\n\n"
        "# Deployment\n\n"
        "The API runs with FastAPI."
    )

    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)

    assert any(
        "# Architecture" in chunk
        and "embeddings and a vector store" in chunk
        for chunk in chunks
    )

    assert any(
        "# Deployment" in chunk
        and "FastAPI" in chunk
        for chunk in chunks
    )


def test_markdown_chunking_respects_maximum_size():
    text = "# Details\n\n" + ("word " * 500)

    chunks = chunk_text(text, chunk_size=200, chunk_overlap=30)

    assert chunks
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_markdown_chunking_preserves_all_sections():
    text = (
        "# A\n\nContent alpha.\n\n"
        "# B\n\nContent beta.\n\n"
        "# C\n\nContent gamma."
    )

    chunks = chunk_text(text, chunk_size=60, chunk_overlap=10)
    combined = "\n".join(chunks)

    assert "Content alpha" in combined
    assert "Content beta" in combined
    assert "Content gamma" in combined



def test_architecture_chunk_contains_component_descriptions():
    from pathlib import Path

    from ai_core.chunking import chunk_text

    readme = Path("README.md").read_text(encoding="utf-8")
    chunks = chunk_text(readme, chunk_size=900, chunk_overlap=150)

    architecture_chunks = [
        chunk
        for chunk in chunks
        if "Main RAG Architecture Components" in chunk
    ]

    assert architecture_chunks, "Architecture section was not chunked."

    component_chunk = next(
        (
            chunk
            for chunk in architecture_chunks
            if "Document Loader" in chunk and "FastAPI" in chunk
        ),
        None,
    )

    assert component_chunk is not None, (
        "Expected a chunk containing the architecture component descriptions."
    )
    assert len(component_chunk) <= 900


def test_rag_returns_fallback_when_hybrid_ranking_is_empty(monkeypatch):
    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(top_k=5)
    assistant.embeddings = SimpleNamespace(
        embed_query=lambda question: [1.0, 0.0]
    )
    assistant.store = SimpleNamespace(
        search=lambda query_vector, top_k: [
            {
                "filename": "notes.txt",
                "text": "Some unrelated information.",
                "score": 0.5,
            }
        ]
    )

    monkeypatch.setattr("ai_core.rag.hybrid_rank", lambda question, candidates: [])

    result = assistant.ask("What is the answer?")

    assert result["sources"] == []
    assert "could not find relevant information" in result["answer"].lower()


def test_rag_uses_custom_top_k(monkeypatch):
    captured = {}

    assistant = object.__new__(KnowledgeAssistant)
    # assistant.settings = SimpleNamespace(top_k=5)
    assistant.settings = SimpleNamespace(top_k=5, openai_api_key="test-key", 
                                         chat_model="mock-model",)
    assistant.embeddings = SimpleNamespace(
        embed_query=lambda question: [1.0, 0.0]
    )

    def fake_search(query_vector, top_k):
        captured["candidate_k"] = top_k
        return [
            {
                "filename": "notes.txt",
                "text": "Relevant information.",
                "score": 1.0,
            }
        ]

    assistant.store = SimpleNamespace(search=fake_search)

    monkeypatch.setattr(
        "ai_core.rag.hybrid_rank",
        lambda question, candidates: candidates,
    )

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content="Relevant answer.")
                        )
                    ]
                ))
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = assistant.ask("What is relevant?", top_k=1)

    assert captured["candidate_k"] == 10
    assert len(result["sources"]) == 1
    assert result["answer"] == "Relevant answer."


def test_rag_passes_question_to_embedding_service():
    captured = {}

    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(top_k=5)

    def fake_embed_query(question):
        captured["question"] = question
        return [1.0, 0.0]

    assistant.embeddings = SimpleNamespace(embed_query=fake_embed_query)
    assistant.store = SimpleNamespace(
        search=lambda query_vector, top_k: []
    )

    assistant.ask("Explain vector retrieval.")

    assert captured["question"] == "Explain vector retrieval."


def test_rag_preserves_retrieved_sources(monkeypatch):
    source = {
        "filename": "guide.md",
        "text": "The system uses retrieval-augmented generation.",
        "score": 0.95,
    }

    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(
        top_k=5,
        openai_api_key="test-key",
        chat_model="mock-model",
    )
    assistant.embeddings = SimpleNamespace(
        embed_query=lambda question: [1.0, 0.0]
    )
    assistant.store = SimpleNamespace(
        search=lambda query_vector, top_k: [source]
    )

    monkeypatch.setattr(
        "ai_core.rag.hybrid_rank",
        lambda question, candidates: candidates,
    )

    class FakeOpenAI:
        def __init__(self, api_key):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(
                                    content="The system uses RAG."
                                )
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    result = assistant.ask("What approach is used?")

    assert result["sources"] == [source]
    assert result["sources"][0]["filename"] == "guide.md"

def test_ingest_returns_indexing_metadata(monkeypatch):
    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(
        chunk_size=100,
        chunk_overlap=10,
    )
    assistant.embeddings = SimpleNamespace(
        embed_texts=lambda chunks: [[1.0, 0.0] for _ in chunks]
    )
    stored = {}
    assistant.store = SimpleNamespace(
        add_document=lambda **kwargs: stored.update(kwargs)
    )

    monkeypatch.setattr(
        "ai_core.rag.load_document",
        lambda filename, content: "Knowledge assistant documentation.",
    )
    monkeypatch.setattr(
        "ai_core.rag.chunk_text",
        lambda text, chunk_size, chunk_overlap: [text],
    )

    result = assistant.ingest("guide.txt", b"document bytes")

    assert result == {
        "filename": "guide.txt",
        "chunks_created": 1,
        "status": "indexed",
    }
    assert stored["filename"] == "guide.txt"
    assert stored["chunks"] == ["Knowledge assistant documentation."]
    assert stored["embeddings"] == [[1.0, 0.0]]


def test_ingest_passes_configured_chunk_settings(monkeypatch):
    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(
        chunk_size=200,
        chunk_overlap=25,
    )
    assistant.embeddings = SimpleNamespace(embed_texts=lambda chunks: [])
    assistant.store = SimpleNamespace(add_document=lambda **kwargs: None)
    captured = {}

    monkeypatch.setattr(
        "ai_core.rag.load_document",
        lambda filename, content: "Some readable document text.",
    )

    def fake_chunk_text(text, chunk_size, chunk_overlap):
        captured["chunk_size"] = chunk_size
        captured["chunk_overlap"] = chunk_overlap
        return ["Some readable document text."]

    monkeypatch.setattr("ai_core.rag.chunk_text", fake_chunk_text)

    assistant.ingest("guide.txt", b"text")

    assert captured == {"chunk_size": 200, "chunk_overlap": 25}


def test_ingest_rejects_whitespace_only_text(monkeypatch):
    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(
        chunk_size=100,
        chunk_overlap=10,
    )
    assistant.embeddings = SimpleNamespace(
        embed_texts=lambda chunks: pytest.fail("Embedding should not run")
    )
    assistant.store = SimpleNamespace(
        add_document=lambda **kwargs: pytest.fail("Storage should not run")
    )

    monkeypatch.setattr(
        "ai_core.rag.load_document",
        lambda filename, content: "   \n\t ",
    )

    with pytest.raises(ValueError, match="no readable text"):
        assistant.ingest("blank.txt", b"content")


def test_ingest_handles_empty_chunk_list(monkeypatch):
    assistant = object.__new__(KnowledgeAssistant)
    assistant.settings = SimpleNamespace(
        chunk_size=100,
        chunk_overlap=10,
    )
    assistant.embeddings = SimpleNamespace(embed_texts=lambda chunks: [])
    stored = {}
    assistant.store = SimpleNamespace(
        add_document=lambda **kwargs: stored.update(kwargs)
    )

    monkeypatch.setattr(
        "ai_core.rag.load_document",
        lambda filename, content: "Readable text that produces no chunks.",
    )
    monkeypatch.setattr(
        "ai_core.rag.chunk_text",
        lambda text, chunk_size, chunk_overlap: [],
    )

    result = assistant.ingest("empty-chunks.txt", b"content")

    assert result["chunks_created"] == 0
    assert stored["chunks"] == []
    assert stored["embeddings"] == []

def test_vector_store_empty_database_returns_no_results(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "empty_vectors.sqlite3"
    monkeypatch.setattr(
        "ai_core.vector_store.get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    store = VectorStore()

    assert store.search([1.0, 0.0]) == []


def test_vector_store_top_k_limits_results(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "top_k_vectors.sqlite3"
    monkeypatch.setattr(
        "ai_core.vector_store.get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    store = VectorStore()
    store.add_document(
        filename="vectors.txt",
        chunks=["Best match", "Second match", "Third match"],
        embeddings=[
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ],
    )

    results = store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2
    assert [item["text"] for item in results] == [
        "Best match",
        "Second match",
    ]


def test_vector_store_skips_embeddings_with_wrong_dimensions(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "dimensions_vectors.sqlite3"
    monkeypatch.setattr(
        "ai_core.vector_store.get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    store = VectorStore()
    store.add_document(
        filename="vectors.txt",
        chunks=["Matching dimensions", "Different dimensions"],
        embeddings=[
            [1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
    )

    results = store.search([1.0, 0.0], top_k=5)

    assert len(results) == 1
    assert results[0]["text"] == "Matching dimensions"


def test_vector_store_lists_documents_and_counts_chunks(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "metadata_vectors.sqlite3"
    monkeypatch.setattr(
        "ai_core.vector_store.get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    store = VectorStore()
    store.add_document(
        filename="b.txt",
        chunks=["Chunk B1", "Chunk B2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    store.add_document(
        filename="a.txt",
        chunks=["Chunk A1"],
        embeddings=[[1.0, 1.0]],
    )

    assert store.count_chunks() == 3
    assert store.list_documents() == [
        {"filename": "a.txt", "chunk_count": 1},
        {"filename": "b.txt", "chunk_count": 2},
    ]

def test_hybrid_rank_lexical_and_phrase_matches_can_improve_ranking():
    results = [
        {"text": "Unrelated content", "score": 0.6},
        {"text": "Vector retrieval system", "score": 0.5},
    ]

    ranked = hybrid_rank("vector retrieval system", results)

    assert ranked[0]["text"] == "Vector retrieval system"
    assert ranked[0]["hybrid_score"] > ranked[1]["hybrid_score"]


def test_hybrid_rank_handles_missing_text_and_score():
    ranked = hybrid_rank("vector search", [{"id": 1}])

    assert len(ranked) == 1
    assert ranked[0]["semantic_score"] == 0.0
    assert ranked[0]["lexical_score"] == 0.0
    assert ranked[0]["phrase_score"] == 0.0
    assert ranked[0]["hybrid_score"] == 0.0


def test_hybrid_rank_preserves_duplicate_candidates():
    candidate = {
        "filename": "guide.md",
        "text": "Vector search uses embeddings.",
        "score": 0.8,
    }

    ranked = hybrid_rank(
        "vector search",
        [candidate.copy(), candidate.copy()],
    )

    assert len(ranked) == 2
    assert all(item["filename"] == "guide.md" for item in ranked)
    assert all("hybrid_score" in item for item in ranked)


def test_hybrid_rank_returns_empty_results_unchanged():
    results = []

    assert hybrid_rank("vector search", results) is results


def test_hybrid_rank_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        hybrid_rank(
            "vector search",
            [{"text": "Vector search", "score": 0.8}],
            semantic_weight=-0.1,
        )

def test_text_loader_supports_utf8_bom():
    content = b"\xef\xbb\xbfHello from UTF-8."

    result = load_document("notes.txt", content)

    assert result == "Hello from UTF-8."


def test_text_loader_rejects_invalid_utf8():
    with pytest.raises(ValueError, match="UTF-8 is required"):
        load_document("notes.txt", b"\xff\xfe\xfa")


def test_text_loader_rejects_whitespace_only_content():
    with pytest.raises(ValueError, match="no usable text"):
        load_document("notes.md", b"   \n\t  ")


def test_pdf_loader_rejects_malformed_content():
    with pytest.raises(ValueError, match="Invalid PDF document"):
        load_document("broken.pdf", b"This is not a PDF file.")


def test_docx_loader_rejects_malformed_content():
    with pytest.raises(ValueError, match="Invalid DOCX document"):
        load_document("broken.docx", b"This is not a DOCX file.")

def test_chunk_text_handles_long_unicode_content():
    text = "你好世界🙂 café " * 100

    chunks = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert "".join(chunks).replace(" ", "")  # Non-empty Unicode output


def test_chunk_text_supports_zero_overlap():
    text = "abcdefghij"

    chunks = chunk_text(
        text,
        chunk_size=5,
        chunk_overlap=0,
    )

    assert chunks == ["abcde", "fghij"]


def test_chunk_text_preserves_heading_in_long_section():
    text = "# Architecture\n\n" + ("Vector retrieval uses embeddings. " * 20)

    chunks = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(chunk.startswith("# Architecture") for chunk in chunks)


def test_chunk_text_respects_chunk_size_for_long_paragraph():
    text = "A" * 500

    chunks = chunk_text(
        text,
        chunk_size=80,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)


def test_chunk_text_preserves_content_across_overlapping_chunks():
    text = "alpha beta gamma delta epsilon zeta eta theta"

    chunks = chunk_text(
        text,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(word in " ".join(chunks) for word in text.split())


