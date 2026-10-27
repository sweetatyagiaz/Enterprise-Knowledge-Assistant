
import pytest

from ai_core.retrieval.ranking import hybrid_rank


def test_hybrid_rank_prefers_relevant_chunk():
    results = [
        {
            "text": "The project uses the following components.",
            "score": 0.66,
        },
        {
            "text": (
                "RAG architecture components include document loader, "
                "text chunking, embeddings, vector store, similarity "
                "search, retrieved context, chat model, and FastAPI."
            ),
            "score": 0.576,
        },
    ]

    ranked = hybrid_rank(
        "What are the main components of this project's RAG architecture?"
        " document loader text chunking embeddings vector store similarity "
        "search retrieved context chat model FastAPI",
        results,
    )

    assert "document loader" in ranked[0]["text"].lower()
    assert ranked[0]["hybrid_score"] > ranked[1]["hybrid_score"]


def test_hybrid_rank_preserves_original_results():
    results = [{"text": "SQLite database", "score": 0.7}]

    ranked = hybrid_rank("SQLite", results)

    assert results[0] == {"text": "SQLite database", "score": 0.7}
    assert ranked[0]["semantic_score"] == pytest.approx(0.7)


def test_hybrid_rank_rejects_invalid_weights():
    with pytest.raises(ValueError):
        hybrid_rank("SQLite", [], semantic_weight=-1)


def test_hybrid_rank_rejects_zero_total_weight():
    with pytest.raises(ValueError):
        hybrid_rank(
            "SQLite",
            [],
            semantic_weight=0,
            lexical_weight=0,
            phrase_weight=0,
        )



def test_hybrid_rank_gives_no_keyword_credit_to_irrelevant_chunks():
    results = [
        {
            "text": "The application uses FastAPI and SQLite for document search.",
            "score": 0.15,
        },
        {
            "text": "Documents are split into overlapping text chunks.",
            "score": 0.12,
        },
    ]

    ranked = hybrid_rank(
        "What is the company's annual profit?",
        results,
    )

    assert all(result["lexical_score"] == 0.0 for result in ranked)
    assert all(result["phrase_score"] == 0.0 for result in ranked)
    assert all("hybrid_score" in result for result in ranked)

def test_hybrid_rank_returns_empty_results_unchanged():
    results = []

    assert hybrid_rank("database search", results) is results


def test_hybrid_rank_preserves_order_when_scores_tie():
    results = [
        {"text": "alpha", "score": 0.5},
        {"text": "beta", "score": 0.5},
    ]

    ranked = hybrid_rank(
        "unrelated query",
        results,
        semantic_weight=1.0,
        lexical_weight=0.0,
        phrase_weight=0.0,
    )

    assert [item["text"] for item in ranked] == ["alpha", "beta"]


def test_hybrid_rank_defaults_missing_semantic_score_to_zero():
    results = [{"text": "database indexing"}]

    ranked = hybrid_rank("database indexing", results)

    assert ranked[0]["semantic_score"] == 0.0
    assert ranked[0]["hybrid_score"] > 0.0


def test_hybrid_rank_returns_original_results_for_stopword_only_query():
    results = [{"text": "database indexing", "score": 0.8}]

    ranked = hybrid_rank("what is the", results)

    assert ranked is results


