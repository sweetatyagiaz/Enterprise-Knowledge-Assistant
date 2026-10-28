"""Evaluate hybrid retrieval and save a timestamped JSON report."""

import json
from datetime import datetime, timezone
from pathlib import Path

from ai_core.embeddings import EmbeddingService
from ai_core.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k as metric_precision_at_k,
    recall_at_k as metric_recall_at_k,
    reciprocal_rank,
)
from ai_core.evaluation.retrieval_report import build_retrieval_report
from ai_core.retrieval.ranking import hybrid_rank
from ai_core.vector_store import VectorStore


DATASET_PATH = Path("tests/data/rag_eval.json")
GROUND_TRUTH_PATH = Path("tests/data/rag_ground_truth.json")
REPORT_DIR = Path("reports/evaluations")
TOP_K = 5
CANDIDATE_K = max(TOP_K, 10)


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    """Check whether all expected keywords appear in the text."""
    text = text.casefold()
    return all(keyword.casefold() in text for keyword in keywords)


def main() -> int:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

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
    case_results = []

    for item in dataset:
        case_id = item["id"]
        question = item["question"]
        expected_source = item.get("expected_source")
        keywords = item.get("expected_keywords", [])
        answerable = item["answerable"]

        query_vector = embeddings.embed_query(question)
        candidates = store.search(
            query_vector,
            top_k=CANDIDATE_K,
        )
        results = hybrid_rank(question, candidates)[:TOP_K]

        texts = [result.get("text", "") for result in results]
        top_1_text = texts[0] if texts else ""
        top_k_text = "\n".join(texts)

        source_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if expected_source
                and result.get("filename") == expected_source
            ),
            None,
        )

        # A missing keyword label means the relevance metric is undefined.
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
            float(contains_all_keywords(top_1_text, keywords))
            if keywords
            else None
        )
        keyword_hit_at_k = (
            float(contains_all_keywords(top_k_text, keywords))
            if keywords
            else None
        )

        reranking_metrics = None

        if keywords:
            # Keyword-based relevance proxy over the initial candidate pool.
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

            reranking_metrics = {
                "precision_at_k": metric_precision_at_k(
                    ranked_ids,
                    relevant_candidate_ids,
                    TOP_K,
                ),
                "recall_at_k": metric_recall_at_k(
                    ranked_ids,
                    relevant_candidate_ids,
                    TOP_K,
                ),
                "mrr": reciprocal_rank(
                    ranked_ids,
                    relevant_candidate_ids,
                ),
                "ndcg_at_k": ndcg_at_k(
                    ranked_ids,
                    candidate_relevance,
                    TOP_K,
                ),
                "candidate_relevant_count": len(relevant_candidate_ids),
            }

        reviewed = reviewed_by_id.get(case_id, {})

        case_result = {
            "id": case_id,
            "question": question,
            "answerable": answerable,
            "expected_source": expected_source,
            "expected_keywords": keywords,
            "ground_truth_reviewed": bool(reviewed),
            "relevant_evidence": reviewed.get("relevant_evidence", []),
            "retrieved_count": len(results),
            "source_rank": source_rank,
            "keyword_evidence_rank": evidence_rank,
            "keyword_evidence_at_1": keyword_hit_at_1,
            "keyword_evidence_at_k": keyword_hit_at_k,
            "reranking_metrics": reranking_metrics,
            "retrieved": [
                {
                    "rank": rank,
                    "id": str(result["id"]),
                    "filename": result.get("filename", "unknown"),
                    "score": result.get(
                        "hybrid_score",
                        result.get("score", 0.0),
                    ),
                    "keyword_relevant_proxy": (
                        bool(
                            contains_all_keywords(
                                result.get("text", ""),
                                keywords,
                            )
                        )
                        if keywords
                        else None
                    ),
                }
                for rank, result in enumerate(results, start=1)
            ],
        }
        case_results.append(case_result)

        print(f"\n[{case_id}] {question}")
        print(f"  Answerable: {answerable}")
        print(f"  Ground truth reviewed: {bool(reviewed)}")
        print(f"  Retrieved chunks: {len(results)}")
        print(f"  Expected source rank: {source_rank}")
        print(f"  Keyword evidence rank: {evidence_rank}")
        print(f"  All keywords found @1: {keyword_hit_at_1}")
        print(f"  All keywords found @{TOP_K}: {keyword_hit_at_k}")

        for result in case_result["retrieved"][:3]:
            print(
                f"  {result['rank']}. {result['filename']} "
                f"(score={result['score']:.3f})"
            )

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
        case["ground_truth_reviewed"] for case in case_results
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / (
        f"retrieval_evaluation_"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]
    print("\n=== Hybrid Retrieval Evaluation ===")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Answerable cases: {summary['answerable_cases']}")
    print(f"Unanswerable cases: {summary['unanswerable_cases']}")
    print(
        "Reviewed ground-truth cases: "
        f"{report['reviewed_ground_truth_cases']}"
    )

    for group_name in ("answerable", "unanswerable"):
        group = summary[group_name]
        print(f"\n{group_name.title()} retrieval metrics ({group['cases']} cases):")

        for metric_name in (
            "source_hit_rate",
            "keyword_evidence_rate_at_1",
            "keyword_evidence_rate_at_k",
            "reranking_precision_at_k",
            "reranking_recall_at_k",
            "reranking_mrr",
            "reranking_ndcg_at_k",
        ):
            metric = group[metric_name]
            print(
                f"  {metric_name}: {metric['mean']} "
                f"(evaluated cases: {metric['evaluated_cases']})"
            )

    print(f"\nJSON report saved: {output_path}")
    print(
        "Reminder: keyword matching is a proxy; reviewed evidence descriptions "
        "are metadata and are not automatically treated as chunk-level labels."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
