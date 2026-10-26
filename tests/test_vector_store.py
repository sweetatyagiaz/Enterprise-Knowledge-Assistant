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
