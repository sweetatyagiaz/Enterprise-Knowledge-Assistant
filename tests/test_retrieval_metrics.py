import math

import pytest

from ai_core.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    assert precision_at_k(
        ["a", "x", "b", "y"],
        {"a", "b"},
        3,
    ) == pytest.approx(2 / 3)


def test_precision_uses_available_results_when_fewer_than_k():
    assert precision_at_k(["a", "x"], {"a"}, 5) == pytest.approx(0.5)


def test_recall_at_k_counts_all_relevant_items():
    assert recall_at_k(
        ["a", "x", "b"],
        {"a", "b", "c"},
        2,
    ) == pytest.approx(1 / 3)


def test_recall_is_zero_when_no_relevant_items_are_known():
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_no_relevant_result_exists():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_is_one_for_ideal_ranking():
    assert ndcg_at_k(
        ["a", "b", "c"],
        {"a": 3.0, "b": 2.0, "c": 1.0},
        3,
    ) == pytest.approx(1.0)


def test_ndcg_penalizes_relevant_items_ranked_lower():
    ideal = ndcg_at_k(
        ["a", "b", "c"],
        {"a": 3.0, "b": 1.0, "c": 0.0},
        3,
    )
    reversed_ranking = ndcg_at_k(
        ["b", "a", "c"],
        {"a": 3.0, "b": 1.0, "c": 0.0},
        3,
    )
    assert reversed_ranking < ideal


def test_empty_results_return_zero():
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert recall_at_k([], {"a"}, 5) == 0.0
    assert reciprocal_rank([], {"a"}) == 0.0
    assert ndcg_at_k([], {"a": 2.0}, 5) == 0.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="positive integer"):
        precision_at_k(["a"], {"a"}, k)


@pytest.mark.parametrize("grade", [-1.0, math.nan, math.inf, -math.inf])
def test_ndcg_rejects_invalid_relevance_grades(grade):
    with pytest.raises(ValueError, match="finite and non-negative"):
        ndcg_at_k(["a"], {"a": grade}, 1)
