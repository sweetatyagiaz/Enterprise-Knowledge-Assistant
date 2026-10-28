
"""Build reproducible JSON reports for retrieval evaluation."""

from datetime import datetime, timezone
from statistics import mean
from typing import Any


def _metric_summary(values: list[float | None]) -> dict[str, Any]:
    """Summarize defined metric values without treating missing values as zero."""
    defined = [value for value in values if value is not None]

    return {
        "evaluated_cases": len(defined),
        "mean": round(mean(defined), 4) if defined else None,
    }


def _group_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize retrieval diagnostics for one answerability group."""
    return {
        "cases": len(cases),
        "source_hit_rate": _metric_summary(
            [
                1.0 if case["source_rank"] is not None else 0.0
                for case in cases
                if case.get("expected_source")
            ]
        ),
        "keyword_evidence_rate_at_1": _metric_summary(
            [
                case["keyword_evidence_at_1"]
                for case in cases
                if case.get("keyword_evidence_at_1") is not None
            ]
        ),
        "keyword_evidence_rate_at_k": _metric_summary(
            [
                case["keyword_evidence_at_k"]
                for case in cases
                if case.get("keyword_evidence_at_k") is not None
            ]
        ),
        "reranking_precision_at_k": _metric_summary(
            [
                case["reranking_metrics"]["precision_at_k"]
                for case in cases
                if case.get("reranking_metrics") is not None
            ]
        ),
        "reranking_recall_at_k": _metric_summary(
            [
                case["reranking_metrics"]["recall_at_k"]
                for case in cases
                if case.get("reranking_metrics") is not None
            ]
        ),
        "reranking_mrr": _metric_summary(
            [
                case["reranking_metrics"]["mrr"]
                for case in cases
                if case.get("reranking_metrics") is not None
            ]
        ),
        "reranking_ndcg_at_k": _metric_summary(
            [
                case["reranking_metrics"]["ndcg_at_k"]
                for case in cases
                if case.get("reranking_metrics") is not None
            ]
        ),
    }


def build_retrieval_report(
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    candidate_k: int,
    dataset_path: str,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable retrieval evaluation report."""
    if top_k < 1 or candidate_k < top_k:
        raise ValueError("Require 1 <= top_k <= candidate_k.")

    answerable = [case for case in cases if case["answerable"]]
    unanswerable = [case for case in cases if not case["answerable"]]

    return {
        "timestamp_utc": (
            timestamp_utc or datetime.now(timezone.utc).isoformat()
        ),
        "dataset": dataset_path,
        "method": "Hybrid retrieval with keyword-based relevance proxies.",
        "configuration": {
            "top_k": top_k,
            "candidate_k": candidate_k,
        },
        "metric_definitions": {
            "source_hit_rate": (
                "Fraction of cases with an expected source where that "
                "source appears in the top-k retrieved results."
            ),
            "keyword_evidence_rate_at_1": (
                "Fraction of labeled cases where the top-ranked passage "
                "contains all expected keywords."
            ),
            "keyword_evidence_rate_at_k": (
                "Fraction of labeled cases where all expected keywords "
                "appear across the combined top-k retrieved text."
            ),
            "reranking_metrics": (
                "Keyword-proxy relevance metrics over the initial candidate "
                "pool; not human-judged relevance."
            ),
        },
        "ground_truth_policy": (
            "Reviewed evidence descriptions are retained as metadata. "
            "They are not automatically converted into chunk-level "
            "relevance labels."
        ),
        "summary": {
            "total_cases": len(cases),
            "answerable_cases": len(answerable),
            "unanswerable_cases": len(unanswerable),
            "answerable": _group_summary(answerable),
            "unanswerable": _group_summary(unanswerable),
        },
        "limitations": [
            "Keyword matching is a relevance proxy, not human-judged relevance.",
            "Reranking recall is relative to the initial candidate pool.",
            "Source hits do not prove that a passage supports the answer.",
            "Retrieval metrics do not measure generated-answer factuality.",
            "Unanswerable-case retrieval statistics do not measure abstention quality.",
            "Undefined metrics are reported as null, not as zero.",
        ],
        "results": cases,
    }
