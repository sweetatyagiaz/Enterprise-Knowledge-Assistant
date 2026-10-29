import json

import pytest

from ai_core.evaluation.reviewed_relevance import (
    evaluate_reviewed_relevance,
    load_relevance_labels,
    validate_chunk_labels,
)


def make_label_record():
    return {
        "case_id": "RAG-001",
        "reviewed": True,
        "relevant_chunks": [
            {
                "chunk_id": 173,
                "grade": 2,
                "evidence": "Main architecture components",
                "evidence_terms": ["Main RAG Architecture Components"],
            },
            {
                "chunk_id": 179,
                "grade": 1,
                "evidence": "Default models",
                "evidence_terms": ["text-embedding-3-small"],
            },
        ],
    }


def test_evaluate_reviewed_relevance_calculates_metrics():
    result = evaluate_reviewed_relevance(
        [173, 200, 179],
        make_label_record(),
        k=3,
    )

    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["ndcg_at_k"] > 0
    assert result["known_relevant_chunk_count"] == 2
    assert result["relevance_judgment"] == "reviewed_chunk_labels"


def test_unreviewed_or_missing_label_returns_none():
    assert evaluate_reviewed_relevance(["173"], None) is None

    record = make_label_record()
    record["reviewed"] = False
    assert evaluate_reviewed_relevance(["173"], record) is None


def test_load_labels_rejects_duplicate_case_ids(tmp_path):
    record = make_label_record()
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([record, record]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate"):
        load_relevance_labels(path)


def test_load_labels_rejects_duplicate_chunk_ids(tmp_path):
    record = make_label_record()
    record["relevant_chunks"].append(record["relevant_chunks"][0])
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate chunk ID"):
        load_relevance_labels(path)


def test_validate_chunk_labels_accepts_matching_chunks():
    labels = {"RAG-001": make_label_record()}
    indexed = [
        {
            "id": 173,
            "filename": "README.md",
            "text": "## Main RAG Architecture Components",
        },
        {
            "id": 179,
            "filename": "README.md",
            "text": "The default embedding model is text-embedding-3-small.",
        },
    ]

    validate_chunk_labels(labels, indexed)


def test_validate_chunk_labels_rejects_missing_chunk():
    labels = {"RAG-001": make_label_record()}

    with pytest.raises(ValueError, match="missing"):
        validate_chunk_labels(labels, [])


def test_validate_chunk_labels_rejects_filename_mismatch():
    labels = {"RAG-001": make_label_record()}
    indexed = [
        {
            "id": 173,
            "filename": "other.md",
            "text": "## Main RAG Architecture Components",
        },
        {
            "id": 179,
            "filename": "README.md",
            "text": "The default embedding model is text-embedding-3-small.",
        },
    ]

    with pytest.raises(ValueError, match="filename mismatch"):
        validate_chunk_labels(labels, indexed)


def test_validate_chunk_labels_rejects_missing_evidence_term():
    labels = {"RAG-001": make_label_record()}
    indexed = [
        {"id": 173, "filename": "README.md", "text": "Unrelated content"},
        {
            "id": 179,
            "filename": "README.md",
            "text": "The default embedding model is text-embedding-3-small.",
        },
    ]

    with pytest.raises(ValueError, match="lacks evidence terms"):
        validate_chunk_labels(labels, indexed)


@pytest.mark.parametrize("grade", [0, -1, True, float("inf")])
def test_load_labels_rejects_invalid_grades(tmp_path, grade):
    record = make_label_record()
    record["relevant_chunks"][0]["grade"] = grade
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    with pytest.raises(ValueError, match="relevance grade"):
        load_relevance_labels(path)
