
import json
from pathlib import Path

from ai_core.embeddings import EmbeddingService
from ai_core.retrieval.ranking import hybrid_rank
from ai_core.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ai_core.vector_store import VectorStore


DATASET_PATH = Path("tests/data/rag_eval.json")
TOP_K = 5
CANDIDATE_K = max(TOP_K, 10)


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    """Check whether all expected keywords appear in the text."""
    text = text.casefold()
    return all(keyword.casefold() in text for keyword in keywords)


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    embeddings = EmbeddingService()
    store = VectorStore()

    answerable_total = 0
    source_hits = 0
    evidence_hits_at_1 = 0
    evidence_hits_at_5 = 0
    keyword_hits_at_1 = 0
    keyword_hits_at_5 = 0
    keyword_test_count = 0
    evidence_reciprocal_ranks = []
    precision_at_k_scores = []
    evidence_hit_at_k_scores = []

    rerank_precision_scores = []
    rerank_recall_scores = []
    rerank_mrr_scores = []
    rerank_ndcg_scores = []

    for item in dataset:
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

        # These relevance labels are keyword-based proxies, not human-
        # judged relevance labels. Recall is relative to the candidate pool.
        if keywords:
            candidate_relevance = {
                str(result["id"]): float(
                    contains_all_keywords(result.get("text", ""), keywords)
                )
                for result in candidates
            }
            relevant_candidate_ids = {
                item_id
                for item_id, grade in candidate_relevance.items()
                if grade > 0
            }
            ranked_ids = [str(result["id"]) for result in results]

            rerank_precision_scores.append(
                precision_at_k(ranked_ids, relevant_candidate_ids, TOP_K)
            )
            rerank_recall_scores.append(
                recall_at_k(ranked_ids, relevant_candidate_ids, TOP_K)
            )
            rerank_mrr_scores.append(
                reciprocal_rank(ranked_ids, relevant_candidate_ids)
            )
            rerank_ndcg_scores.append(
                ndcg_at_k(ranked_ids, candidate_relevance, TOP_K)
            )

        texts = [result.get("text", "") for result in results]
        top_1_text = texts[0] if texts else ""
        top_5_text = "\n".join(texts)

        source_rank = next(
            (
                rank
                for rank, result in enumerate(results, start=1)
                if expected_source
                and result.get("filename") == expected_source
            ),
            None,
        )

        # Rank of the first individual passage containing all keywords.
        relevant_at_k = [
            bool(keywords and contains_all_keywords(text, keywords))
            for text in texts
        ]

        relevant_count_at_k = sum(relevant_at_k)

        evidence_rank = next(
            (
                rank
                for rank, is_relevant in enumerate(relevant_at_k, start=1)
                if is_relevant
            ),
            None,
        )

        keyword_hit_at_1 = (
            contains_all_keywords(top_1_text, keywords)
            if keywords else None
        )
        keyword_hit_at_5 = (
            contains_all_keywords(top_5_text, keywords)
            if keywords else None
        )

        if answerable:
            answerable_total += 1
            source_hits += int(source_rank is not None)

            evidence_hits_at_1 += int(keyword_hit_at_1 is True)
            evidence_hits_at_5 += int(keyword_hit_at_5 is True)

            precision_at_k = (
                relevant_count_at_k / len(results)
                if results else 0.0
            )

            recall_at_k = (
                1.0 if evidence_rank is not None else 0.0
            )

            precision_at_k_scores.append(precision_at_k)
            evidence_hit_at_k_scores.append(recall_at_k)

            evidence_reciprocal_ranks.append(
                1 / evidence_rank if evidence_rank is not None else 0.0
            )

            if keywords:
                keyword_test_count += 1
                keyword_hits_at_1 += int(keyword_hit_at_1)
                keyword_hits_at_5 += int(keyword_hit_at_5)

        print(f"\n[{item['id']}] {question}")
        print(f"  Answerable: {answerable}")
        print(f"  Retrieved chunks: {len(results)}")
        print(f"  Expected source rank: {source_rank}")
        print(f"  All keywords in one passage: {evidence_rank is not None}")
        print(f"  Keyword evidence rank: {evidence_rank}")
        print(f"  All keywords found @1: {keyword_hit_at_1}")
        print(f"  All keywords found @{TOP_K}: {keyword_hit_at_5}")

        for rank, result in enumerate(results[:3], start=1):
            score = result.get(
                "hybrid_score",
                result.get("score", 0.0),
            )
            print(
                f"  {rank}. {result.get('filename', 'unknown')} "
                f"(score={score:.3f})"
            )

    print("\n=== Hybrid Retrieval Evaluation ===")
    print(f"Total cases: {len(dataset)}")
    print(f"Answerable cases: {answerable_total}")
    print(f"Unanswerable cases: {len(dataset) - answerable_total}")

    if answerable_total:
        print(
            f"Source hit rate @{TOP_K}: "
            f"{source_hits / answerable_total:.1%}"
        )
        print(
            "Passage evidence rate @1: "
            f"{evidence_hits_at_1 / answerable_total:.1%}"
        )
        print(
            f"Passage evidence rate @{TOP_K}: "
            f"{evidence_hits_at_5 / answerable_total:.1%}"
        )
        print(
            "Evidence MRR: "
            f"{sum(evidence_reciprocal_ranks) / answerable_total:.3f}"
        )
        if precision_at_k_scores:
            print(
                f"Keyword-based Precision@{TOP_K}: "
                f"{sum(precision_at_k_scores) / len(precision_at_k_scores):.1%}"
            )

        if evidence_hit_at_k_scores:
            print(
                f"Keyword evidence hit rate@{TOP_K}: "
                f"{sum(evidence_hit_at_k_scores) / len(evidence_hit_at_k_scores):.1%}"
            )

        if rerank_precision_scores:
            print(
                f"Keyword-proxy reranking Precision@{TOP_K}: "
                f"{sum(rerank_precision_scores) / len(rerank_precision_scores):.1%}"
            )
            print(
                f"Keyword-proxy candidate-pool Recall@{TOP_K}: "
                f"{sum(rerank_recall_scores) / len(rerank_recall_scores):.1%}"
            )
            print(
                f"Keyword-proxy reranking MRR: "
                f"{sum(rerank_mrr_scores) / len(rerank_mrr_scores):.3f}"
            )
            print(
                f"Keyword-proxy reranking nDCG@{TOP_K}: "
                f"{sum(rerank_ndcg_scores) / len(rerank_ndcg_scores):.3f}"
            )

        if keyword_test_count:
            print(
                f"Keyword coverage @1: "
                f"{keyword_hits_at_1 / keyword_test_count:.1%}"
            )
            print(
                f"Keyword coverage @{TOP_K}: "
                f"{keyword_hits_at_5 / keyword_test_count:.1%}"
            )

    print(
        "\nNote: keyword-proxy reranking metrics use the initial candidate "
        "pool as their relevance universe. They do not measure recall across "
        "the full database or replace human-judged relevance labels. These "
        "metrics do not measure generated-answer factuality. Unanswerable "
        "cases require separate abstention evaluation."
    )


if __name__ == "__main__":
    main()
