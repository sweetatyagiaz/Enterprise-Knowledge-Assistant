"""Ranking metrics for evaluating information retrieval quality."""

import math
from collections.abc import Sequence


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")


def precision_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return relevant results divided by the requested cutoff k."""
    _validate_k(k)

    if not ranked_ids:
        return 0.0

    retrieved = ranked_ids[:k]
    hits = sum(item in relevant_ids for item in retrieved)
    return hits / min(k, len(retrieved)) if retrieved else 0.0


def recall_at_k(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Return the fraction of known relevant items found in the top k."""
    _validate_k(k)

    if not relevant_ids:
        return 0.0

    hits = sum(item in relevant_ids for item in ranked_ids[:k])
    return hits / len(relevant_ids)


def reciprocal_rank(
    ranked_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for rank, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    """Return normalized discounted cumulative gain at k.

    Relevance grades must be finite and non-negative. Gains use the
    conventional exponential gain 2**relevance - 1.
    """
    _validate_k(k)

    for grade in relevance.values():
        if not math.isfinite(grade) or grade < 0:
            raise ValueError("Relevance grades must be finite and non-negative.")

    def dcg(grades: Sequence[float]) -> float:
        return sum(
            (2**grade - 1) / math.log2(rank + 1)
            for rank, grade in enumerate(grades, start=1)
        )

    actual = [relevance.get(item, 0.0) for item in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]

    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0

    return dcg(actual) / ideal_dcg
