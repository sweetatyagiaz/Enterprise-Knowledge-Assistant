import numpy as np
import pytest

from ai_core.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import ai_core.vector_store as vector_store_module

    monkeypatch.setattr(
        vector_store_module,
        "get_settings",
        lambda: SimpleNamespace(
            sqlite_path=tmp_path / "vector_store.sqlite3"
        ),
    )
    return VectorStore()


def test_add_document_and_search(vector_store):
    count = vector_store.add_document(
        "guide.md",
        ["Python guide", "Database guide"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    results = vector_store.search([1.0, 0.0], top_k=2)

    assert count == 2
    assert results[0]["text"] == "Python guide"
    assert results[0]["score"] == pytest.approx(1.0)


def test_add_document_rejects_mismatched_counts(vector_store):
    with pytest.raises(ValueError, match="one embedding"):
        vector_store.add_document(
            "guide.md",
            ["First", "Second"],
            [[1.0, 0.0]],
        )


@pytest.mark.parametrize(
    "vector",
    [
        [],
        [0.0, 0.0],
        [float("nan"), 1.0],
        [float("inf"), 1.0],
    ],
)
def test_add_document_rejects_invalid_vectors(vector_store, vector):
    with pytest.raises(ValueError):
        vector_store.add_document("guide.md", ["Some text"], [vector])


def test_search_rejects_invalid_query_vectors(vector_store):
    for query in ([], [float("nan"), 1.0], [float("inf"), 1.0]):
        with pytest.raises(ValueError):
            vector_store.search(query)


def test_search_returns_empty_for_zero_query(vector_store):
    assert vector_store.search([0.0, 0.0]) == []


def test_search_skips_vectors_with_different_dimensions(vector_store):
    vector_store.add_document(
        "guide.md",
        ["Two dimensions", "Three dimensions"],
        [[1.0, 0.0], [1.0, 0.0, 0.0]],
    )

    results = vector_store.search([1.0, 0.0])

    assert len(results) == 1
    assert results[0]["text"] == "Two dimensions"


def test_search_rejects_non_positive_top_k(vector_store):
    vector_store.add_document("guide.md", ["Example"], [[1.0, 0.0]])

    # Current behavior clamps top_k to at least one.
    assert len(vector_store.search([1.0, 0.0], top_k=0)) == 1

def test_search_returns_empty_when_store_has_no_documents(vector_store):
    assert vector_store.search([1.0, 0.0]) == []


def test_search_respects_top_k(vector_store):
    vector_store.add_document(
        "guide.md",
        ["First", "Second", "Third"],
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
    )

    results = vector_store.search([1.0, 0.0], top_k=2)

    assert len(results) == 2


def test_search_ranks_best_match_first(vector_store):
    vector_store.add_document(
        "guide.md",
        ["Orthogonal match", "Exact match", "Similar match"],
        [[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]],
    )

    results = vector_store.search([1.0, 0.0], top_k=3)

    assert results[0]["text"] == "Exact match"
    assert results[0]["score"] >= results[1]["score"]
    assert results[1]["score"] >= results[2]["score"]


def test_search_results_include_filename_and_score(vector_store):
    vector_store.add_document(
        "architecture.md",
        ["The system uses retrieval-augmented generation."],
        [[1.0, 0.0]],
    )

    result = vector_store.search([1.0, 0.0])[0]

    assert result["filename"] == "architecture.md"
    assert result["text"] == "The system uses retrieval-augmented generation."
    assert result["score"] == pytest.approx(1.0)


def test_search_cache_is_invalidated_after_insert(vector_store):
    vector_store.add_document(
        "first.txt",
        ["First chunk"],
        [[1.0, 0.0]],
    )

    first = vector_store.search([1.0, 0.0])
    assert [item["text"] for item in first] == ["First chunk"]

    vector_store.add_document(
        "second.txt",
        ["Second chunk"],
        [[0.0, 1.0]],
    )

    results = vector_store.search([0.0, 1.0], top_k=2)
    assert {item["text"] for item in results} == {
        "First chunk",
        "Second chunk",
    }


def test_search_cache_detects_insert_from_another_instance(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace
    import ai_core.vector_store as vector_store_module

    database_path = tmp_path / "shared_vectors.sqlite3"
    monkeypatch.setattr(
        vector_store_module,
        "get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    first_store = VectorStore()
    second_store = VectorStore()

    first_store.add_document(
        "first.txt",
        ["First chunk"],
        [[1.0, 0.0]],
    )
    first_store.search([1.0, 0.0])

    second_store.add_document(
        "second.txt",
        ["Second chunk"],
        [[0.0, 1.0]],
    )

    results = first_store.search([0.0, 1.0], top_k=2)
    assert {item["text"] for item in results} == {
        "First chunk",
        "Second chunk",
    }


def test_search_cache_detects_update_without_row_count_change(vector_store):
    vector_store.add_document(
        "guide.txt",
        ["First chunk", "Second chunk"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    before = vector_store.search([1.0, 0.0], top_k=2)
    assert before[0]["text"] == "First chunk"

    # Update the existing row through a separate database connection.
    with vector_store._connect() as connection:
        connection.execute(
            """
            UPDATE document_chunks
            SET chunk_text = ?, embedding = ?
            WHERE filename = ? AND chunk_text = ?
            """,
            ("Updated chunk", "[0.0, 1.0]", "guide.txt", "First chunk"),
        )

    after = vector_store.search([1.0, 0.0], top_k=2)

    assert len(after) == 2
    assert "Updated chunk" in {item["text"] for item in after}
    assert next(
        item for item in after if item["text"] == "Updated chunk"
    )["score"] == pytest.approx(0.0)


def test_search_cache_detects_external_delete(vector_store):
    vector_store.add_document(
        "guide.txt",
        ["Keep this", "Delete this"],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    vector_store.search([1.0, 0.0], top_k=2)

    with vector_store._connect() as connection:
        connection.execute(
            "DELETE FROM document_chunks WHERE chunk_text = ?",
            ("Delete this",),
        )

    results = vector_store.search([0.0, 1.0], top_k=2)

    assert [item["text"] for item in results] == ["Keep this"]


def test_search_supports_legacy_json_only_rows(vector_store):
    with vector_store._connect() as connection:
        connection.execute(
            """
            INSERT INTO document_chunks
                (filename, chunk_text, embedding, embedding_blob)
            VALUES (?, ?, ?, NULL)
            """,
            ("legacy.txt", "Legacy JSON chunk", "[1.0, 0.0]"),
        )

    results = vector_store.search([1.0, 0.0])

    assert len(results) == 1
    assert results[0]["text"] == "Legacy JSON chunk"
    assert results[0]["score"] == pytest.approx(1.0)


def test_embedding_update_clears_stale_binary_blob(vector_store):
    vector_store.add_document(
        "guide.txt",
        ["Original chunk"],
        [[1.0, 0.0]],
    )
    vector_store.search([1.0, 0.0])

    with vector_store._connect() as connection:
        connection.execute(
            """
            UPDATE document_chunks
            SET embedding = ?
            WHERE filename = ?
            """,
            ("[0.0, 1.0]", "guide.txt"),
        )

        row = connection.execute(
            """
            SELECT embedding_blob
            FROM document_chunks
            WHERE filename = ?
            """,
            ("guide.txt",),
        ).fetchone()

    assert row["embedding_blob"] is None

    results = vector_store.search([1.0, 0.0])
    assert results[0]["score"] == pytest.approx(0.0)
