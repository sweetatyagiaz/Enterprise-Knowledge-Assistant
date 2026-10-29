
"""Evaluate hybrid retrieval and save a timestamped JSON report."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_core.embeddings import EmbeddingService
from ai_core.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k as metric_precision_at_k,
    recall_at_k as metric_recall_at_k,
    reciprocal_rank,
)
from ai_core.evaluation.reviewed_relevance import (
    evaluate_reviewed_relevance,
    load_relevance_labels,
    validate_chunk_labels,
)
from ai_core.evaluation.retrieval_report import build_retrieval_report
from ai_core.retrieval.ranking import hybrid_rank
from ai_core.vector_store import VectorStore


DATASET_PATH = Path("tests/data/rag_eval.json")
GROUND_TRUTH_PATH = Path("tests/data/rag_ground_truth.json")
RELEVANCE_LABELS_PATH = Path(
    "tests/data/retrieval_relevance_labels.json"
)
REPORT_DIR = Path("reports/evaluations")

TOP_K = 5
CANDIDATE_K = max(TOP_K, 10)


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    """Check whether all expected keywords appear in the text."""
    text = text.casefold()
    return all(keyword.casefold() in text for keyword in keywords)


def load_reviewed_labels(
    store: VectorStore,
    labels_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load reviewed labels and validate them against the current index."""
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Reviewed relevance labels not found: {labels_path}"
        )

    relevance_labels = load_relevance_labels(labels_path)

    with store._connect() as connection:
        rows = connection.execute(
            """
            SELECT id, filename, chunk_text
            FROM document_chunks
            """
        ).fetchall()

    indexed_chunks = [
        {
            "id": row["id"],
            "filename": row["filename"],
            "text": row["chunk_text"],
        }
        for row in rows
    ]

    validate_chunk_labels(relevance_labels, indexed_chunks)

    print(f"Reviewed relevance cases: {len(relevance_labels)}")
    print(
        "Reviewed relevance labels: "
        f"{sum(len(item['relevant_chunks']) for item in relevance_labels.values())}"
    )
    print(f"Indexed chunks validated: {len(indexed_chunks)}")

    return relevance_labels


def calculate_keyword_proxy_metrics(
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    keywords: list[str],
    top_k: int,
) -> dict[str, Any] | None:
    """Calculate existing keyword-based relevance proxy metrics."""
    if not keywords:
        return None

    candidate_relevance = {
        str(result["id"]): float(
            contains_all_keywords(
                result.get("text", ""),
                keywords,
            )
        )
        for result in candidates
    }

    relevant_candidate_ids = {
        result_id
        for result_id, grade in candidate_relevance.items()
        if grade > 0
    }

    ranked_ids = [str(result["id"]) for result in results]

    return {
        "precision_at_k": metric_precision_at_k(
            ranked_ids,
            relevant_candidate_ids,
            top_k,
        ),
        "recall_at_k": metric_recall_at_k(
            ranked_ids,
            relevant_candidate_ids,
            top_k,
        ),
        "mrr": reciprocal_rank(
            ranked_ids,
            relevant_candidate_ids,
        ),
        "ndcg_at_k": ndcg_at_k(
            ranked_ids,
            candidate_relevance,
            top_k,
        ),
        "candidate_relevant_count": len(relevant_candidate_ids),
    }


def build_retrieved_chunk_details(
    results: list[dict[str, Any]],
    keywords: list[str],
    label_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build ranked retrieval details, including reviewed labels when available."""
    reviewed_grades = {}

    if label_record is not None:
        reviewed_grades = {
            str(item["chunk_id"]): item["grade"]
            for item in label_record["relevant_chunks"]
        }

    retrieved = []

    for rank, result in enumerate(results, start=1):
        chunk_id = str(result["id"])
        text = result.get("text", "")

        retrieved_item = {
            "rank": rank,
            "id": chunk_id,
            "filename": result.get("filename", "unknown"),
            "score": result.get(
                "hybrid_score",
                result.get("score", 0.0),
            ),
            "keyword_relevant_proxy": (
                contains_all_keywords(text, keywords)
                if keywords
                else None
            ),
            "reviewed_relevance_grade": (
                reviewed_grades.get(chunk_id)
                if label_record is not None
                else None
            ),
        }

        retrieved.append(retrieved_item)

    return retrieved


def main() -> int:
    """Run retrieval evaluation and save the JSON report."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    ground_truth = []
    if GROUND_TRUTH_PATH.exists():
        ground_truth = json.loads(
            GROUND_TRUTH_PATH.read_text(encoding="utf-8")
        )

    reviewed_by_id = {
        item["id"]: item
        for item in ground_truth
        if item.get("reviewed") is True
    }

    embeddings = EmbeddingService()
    store = VectorStore()

    # The reviewed-label dataset is intentionally local and is not
    # committed to GitHub. Validate it before evaluating any questions.
    relevance_labels = load_reviewed_labels(
        store,
        RELEVANCE_LABELS_PATH,
    )

    case_results = []

    for item in dataset:
        case_id = item["id"]
        question = item["question"]
        expected_source = item.get("expected_source")
        keywords = item.get("expected_keywords", [])
        answerable = item["answerable"]

        # Retrieve candidates and apply the existing hybrid ranking.
        query_vector = embeddings.embed_query(question)

        candidates = store.search(
            query_vector,
            top_k=CANDIDATE_K,
        )

        results = hybrid_rank(
            question,
            candidates,
        )[:TOP_K]

        texts = [
            result.get("text", "")
            for result in results
        ]

        top_1_text = texts[0] if texts else ""
        top_k_text = "\n".join(texts)

        # Find the rank of the expected source, if one is specified.
        source_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if expected_source
                and result.get("filename") == expected_source
            ),
            None,
        )

        # Existing keyword evidence diagnostics.
        relevant_at_k = (
            [
                contains_all_keywords(text, keywords)
                for text in texts
            ]
            if keywords
            else []
        )

        evidence_rank = next(
            (
                rank
                for rank, relevant in enumerate(relevant_at_k, start=1)
                if relevant
            ),
            None,
        )

        keyword_hit_at_1 = (
            float(
                contains_all_keywords(
                    top_1_text,
                    keywords,
                )
            )
            if keywords
            else None
        )

        keyword_hit_at_k = (
            float(
                contains_all_keywords(
                    top_k_text,
                    keywords,
                )
            )
            if keywords
            else None
        )

        # Preserve the existing keyword-proxy reranking metrics.
        reranking_metrics = calculate_keyword_proxy_metrics(
            candidates=candidates,
            results=results,
            keywords=keywords,
            top_k=TOP_K,
        )

        # Existing reviewed answer-ground-truth metadata.
        reviewed = reviewed_by_id.get(case_id, {})

        # Separate human-reviewed chunk relevance evaluation.
        # Unanswerable cases are not assigned relevance metrics.
        label_record = (
            relevance_labels.get(case_id)
            if answerable
            else None
        )

        reviewed_relevance_metrics = evaluate_reviewed_relevance(
            [result["id"] for result in results],
            label_record,
            TOP_K,
        )

        retrieved_details = build_retrieved_chunk_details(
            results=results,
            keywords=keywords,
            label_record=label_record,
        )

        case_result = {
            "id": case_id,
            "question": question,
            "answerable": answerable,
            "expected_source": expected_source,
            "expected_keywords": keywords,
            "ground_truth_reviewed": bool(reviewed),
            "relevant_evidence": reviewed.get(
                "relevant_evidence",
                [],
            ),
            "retrieved_count": len(results),
            "source_rank": source_rank,
            "keyword_evidence_rank": evidence_rank,
            "keyword_evidence_at_1": keyword_hit_at_1,
            "keyword_evidence_at_k": keyword_hit_at_k,
            "reranking_metrics": reranking_metrics,
            "reviewed_relevance_metrics": reviewed_relevance_metrics,
            "retrieved": retrieved_details,
        }

        case_results.append(case_result)

        # Per-question console output.
        print(f"\n[{case_id}] {question}")
        print(f"  Answerable: {answerable}")
        print(f"  Ground truth reviewed: {bool(reviewed)}")
        print(
            "  Reviewed chunk labels available: "
            f"{label_record is not None}"
        )
        print(f"  Retrieved chunks: {len(results)}")
        print(f"  Expected source rank: {source_rank}")
        print(f"  Keyword evidence rank: {evidence_rank}")
        print(f"  All keywords found @1: {keyword_hit_at_1}")
        print(f"  All keywords found @{TOP_K}: {keyword_hit_at_k}")

        if reviewed_relevance_metrics is not None:
            print(
                "  Reviewed Precision@K: "
                f"{reviewed_relevance_metrics['precision_at_k']}"
            )
            print(
                "  Reviewed Recall@K: "
                f"{reviewed_relevance_metrics['recall_at_k']}"
            )
            print(
                "  Reviewed MRR: "
                f"{reviewed_relevance_metrics['mrr']}"
            )
            print(
                "  Reviewed NDCG@K: "
                f"{reviewed_relevance_metrics['ndcg_at_k']}"
            )

        for result in retrieved_details[:3]:
            print(
                f"  {result['rank']}. {result['filename']} "
                f"(id={result['id']}, score={result['score']:.3f})"
            )

    # Build a timestamped report.
    timestamp = datetime.now(timezone.utc)

    report = build_retrieval_report(
        case_results,
        top_k=TOP_K,
        candidate_k=CANDIDATE_K,
        dataset_path=str(DATASET_PATH),
        timestamp_utc=timestamp.isoformat(),
    )

    report["ground_truth_dataset"] = str(GROUND_TRUTH_PATH)
    report["reviewed_ground_truth_cases"] = sum(
        case["ground_truth_reviewed"]
        for case in case_results
    )
    report["reviewed_relevance_dataset"] = str(
        RELEVANCE_LABELS_PATH
    )
    report["reviewed_relevance_label_cases"] = len(
        relevance_labels
    )
    report["reviewed_relevance_label_count"] = sum(
        len(item["relevant_chunks"])
        for item in relevance_labels.values()
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORT_DIR / (
        "retrieval_evaluation_"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    # Print aggregate results.
    summary = report["summary"]

    print("\n=== Hybrid Retrieval Evaluation ===")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Answerable cases: {summary['answerable_cases']}")
    print(f"Unanswerable cases: {summary['unanswerable_cases']}")
    print(
        "Reviewed ground-truth cases: "
        f"{report['reviewed_ground_truth_cases']}"
    )
    print(
        "Reviewed relevance-label cases: "
        f"{report['reviewed_relevance_label_cases']}"
    )
    print(
        "Reviewed relevance-label count: "
        f"{report['reviewed_relevance_label_count']}"
    )

    keyword_metric_names = (
        "source_hit_rate",
        "keyword_evidence_rate_at_1",
        "keyword_evidence_rate_at_k",
        "reranking_precision_at_k",
        "reranking_recall_at_k",
        "reranking_mrr",
        "reranking_ndcg_at_k",
    )

    reviewed_metric_names = (
        "reviewed_precision_at_k",
        "reviewed_recall_at_k",
        "reviewed_mrr",
        "reviewed_ndcg_at_k",
    )

    for group_name in ("answerable", "unanswerable"):
        group = summary[group_name]

        print(
            f"\n{group_name.title()} retrieval metrics "
            f"({group['cases']} cases):"
        )

        for metric_name in keyword_metric_names:
            metric = group[metric_name]

            print(
                f"  {metric_name}: {metric['mean']} "
                f"(evaluated cases: {metric['evaluated_cases']})"
            )

        print(
            "\n  Human-reviewed relevance metrics "
            f"({group['reviewed_relevance_cases']} labeled cases):"
        )

        for metric_name in reviewed_metric_names:
            metric = group[metric_name]

            print(
                f"    {metric_name}: {metric['mean']} "
                f"(evaluated cases: {metric['evaluated_cases']})"
            )

    print(f"\nJSON report saved: {output_path}")
    print(
        "Reminder: keyword matching is a proxy. Reviewed chunk-level "
        "metrics use the separate local relevance-label dataset."
    )
    print(
        "Reviewed recall is relative to known labeled relevant chunks; "
        "the labels may not cover every relevant chunk in the corpus."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
