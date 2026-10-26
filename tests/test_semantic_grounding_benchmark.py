import json

import pytest

from tests.evaluate_semantic_grounding_adversarial import (
    DATASET_PATH,
    VERDICTS,
    calculate_classification_metrics,
)


def test_benchmark_contains_all_three_verdicts():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    expected_verdicts = {
        item["expected_verdict"] for item in dataset
    }

    assert len(dataset) == 18
    assert expected_verdicts == set(VERDICTS)


def test_benchmark_ids_are_unique():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    ids = [item["id"] for item in dataset]

    assert len(ids) == len(set(ids))


def test_benchmark_cases_have_required_fields():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    required_fields = {
        "id",
        "question",
        "answer",
        "evidence",
        "expected_verdict",
    }

    for item in dataset:
        assert required_fields.issubset(item.keys())
        assert item["expected_verdict"] in VERDICTS
        assert item["question"].strip()
        assert item["answer"].strip()
        assert item["evidence"].strip()


def test_confusion_matrix_and_classification_metrics():
    results = [
        {
            "expected_verdict": "SUPPORTED",
            "actual_verdict": "SUPPORTED",
        },
        {
            "expected_verdict": "SUPPORTED",
            "actual_verdict": "CONTRADICTED",
        },
        {
            "expected_verdict": "CONTRADICTED",
            "actual_verdict": "CONTRADICTED",
        },
        {
            "expected_verdict": "NOT_ENOUGH_EVIDENCE",
            "actual_verdict": "NOT_ENOUGH_EVIDENCE",
        },
    ]

    matrix, metrics = calculate_classification_metrics(results)

    assert matrix["SUPPORTED"]["SUPPORTED"] == 1
    assert matrix["SUPPORTED"]["CONTRADICTED"] == 1
    assert matrix["CONTRADICTED"]["CONTRADICTED"] == 1
    assert matrix["NOT_ENOUGH_EVIDENCE"]["NOT_ENOUGH_EVIDENCE"] == 1

    assert metrics["SUPPORTED"]["precision"] == pytest.approx(1.0)
    assert metrics["SUPPORTED"]["recall"] == pytest.approx(0.5)
    assert metrics["SUPPORTED"]["f1"] == pytest.approx(2 / 3)

    assert metrics["CONTRADICTED"]["precision"] == pytest.approx(0.5)
    assert metrics["CONTRADICTED"]["recall"] == pytest.approx(1.0)
    assert metrics["CONTRADICTED"]["f1"] == pytest.approx(2 / 3)

    assert metrics["NOT_ENOUGH_EVIDENCE"]["precision"] == pytest.approx(1.0)
    assert metrics["NOT_ENOUGH_EVIDENCE"]["recall"] == pytest.approx(1.0)
    assert metrics["NOT_ENOUGH_EVIDENCE"]["f1"] == pytest.approx(1.0)


def test_metrics_handle_empty_results():
    matrix, metrics = calculate_classification_metrics([])

    for verdict in VERDICTS:
        assert sum(matrix[verdict].values()) == 0
        assert metrics[verdict]["support"] == 0
        assert metrics[verdict]["precision"] is None
        assert metrics[verdict]["recall"] is None
        assert metrics[verdict]["f1"] is None


def test_missing_evidence_is_not_automatically_a_contradiction():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    case = next(item for item in dataset if item["id"] == "ADV-010")

    assert case["expected_verdict"] == "NOT_ENOUGH_EVIDENCE"
    assert "executable files" in case["answer"].lower()
    assert "explicitly prohibited" not in case["evidence"].lower()


def test_explicit_prohibition_supports_contradiction_label():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    case = next(item for item in dataset if item["id"] == "ADV-011")

    assert case["expected_verdict"] == "CONTRADICTED"
    assert "explicitly prohibited" in case["evidence"].lower()
