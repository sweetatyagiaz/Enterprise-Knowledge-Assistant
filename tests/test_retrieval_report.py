import json

import pytest

from ai_core.evaluation.retrieval_report import build_retrieval_report


def make_case(
    case_id,
    *,
    answerable,
    expected_source=None,
    source_rank=None,
    keyword_evidence_at_1=None,
    keyword_evidence_at_k=None,
    reranking_metrics=None,
):
    return {
        "id": case_id,
        "question": f"Question {case_id}",
        "answerable": answerable,
        "expected_source": expected_source,
        "source_rank": source_rank,
        "keyword_evidence_at_1": keyword_evidence_at_1,
        "keyword_evidence_at_k": keyword_evidence_at_k,
        "reranking_metrics": reranking_metrics,
    }


def test_report_separates_answerable_and_unanswerable_cases():
    report = build_retrieval_report(
        [
            make_case(
                "RAG-001",
                answerable=True,
                expected_source="README.md",
                source_rank=1,
                keyword_evidence_at_1=1.0,
                keyword_evidence_at_k=1.0,
            ),
            make_case("RAG-005", answerable=False),
        ],
        top_k=5,
        candidate_k=10,
        dataset_path="tests/data/rag_eval.json",
        timestamp_utc="2026-01-01T00:00:00+00:00",
    )

    summary = report["summary"]

    assert summary["total_cases"] == 2
    assert summary["answerable_cases"] == 1
    assert summary["unanswerable_cases"] == 1
    assert summary["answerable"]["source_hit_rate"]["mean"] == 1.0
    assert summary["unanswerable"]["source_hit_rate"]["evaluated_cases"] == 0


def test_missing_keyword_metrics_are_not_counted_as_zero():
    report = build_retrieval_report(
        [make_case("RAG-005", answerable=False)],
        top_k=5,
        candidate_k=10,
        dataset_path="tests/data/rag_eval.json",
    )

    metric = report["summary"]["unanswerable"]["keyword_evidence_rate_at_k"]

    assert metric == {"evaluated_cases": 0, "mean": None}


def test_report_is_json_serializable():
    report = build_retrieval_report(
        [make_case("RAG-001", answerable=True)],
        top_k=5,
        candidate_k=10,
        dataset_path="tests/data/rag_eval.json",
    )

    restored = json.loads(json.dumps(report))

    assert restored["results"][0]["id"] == "RAG-001"
    assert restored["configuration"]["top_k"] == 5


@pytest.mark.parametrize(
    ("top_k", "candidate_k"),
    [(0, 10), (5, 4), (-1, 10)],
)
def test_report_rejects_invalid_cutoffs(top_k, candidate_k):
    with pytest.raises(ValueError):
        build_retrieval_report(
            [],
            top_k=top_k,
            candidate_k=candidate_k,
            dataset_path="tests/data/rag_eval.json",
        )



def test_report_documents_keyword_metric_semantics_and_ground_truth_policy():
    report = build_retrieval_report(
        [],
        top_k=5,
        candidate_k=10,
        dataset_path="tests/data/rag_eval.json",
    )

    definitions = report["metric_definitions"]

    assert "combined top-k retrieved text" in (
        definitions["keyword_evidence_rate_at_k"]
    )
    assert "not human-judged relevance" in (
        definitions["reranking_metrics"]
    )
    assert "not automatically converted" in report["ground_truth_policy"]
