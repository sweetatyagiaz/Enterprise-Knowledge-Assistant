
import json
from pathlib import Path

from ai_core.embeddings import EmbeddingService
from ai_core.retrieval.ranking import hybrid_rank
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
        "\nNote: these are keyword-based evidence proxies, not a measure "
        "of generated-answer factuality. Unanswerable cases require a "
        "separate test of whether the assistant correctly abstains."
    )


if __name__ == "__main__":
    main()
