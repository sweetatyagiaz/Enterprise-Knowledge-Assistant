from tests.evaluate_semantic_grounding import evaluate_answer_quality


def test_answerable_case_with_expected_keywords():
    reference = {
        "answerable": True,
        "reference_answer": "The system uses SQLite for vector storage.",
        "expected_keywords": ["SQLite", "vector storage"],
    }

    result = evaluate_answer_quality(
        "The system uses SQLite for vector storage.",
        reference,
    )

    assert result["answer_check"]["all_keywords_in_answer"] is True
    assert result["answer_check"]["behavior_check_passed"] is True
    assert result["reference_answer_comparison"]["f1"] == 1.0


def test_answerable_case_missing_expected_keyword():
    reference = {
        "answerable": True,
        "reference_answer": "The system uses SQLite for vector storage.",
        "expected_keywords": ["SQLite", "vector storage"],
    }

    result = evaluate_answer_quality(
        "The system uses SQLite.",
        reference,
    )

    assert result["answer_check"]["all_keywords_in_answer"] is False
    assert result["answer_check"]["behavior_check_passed"] is False


def test_unanswerable_case_with_abstention():
    reference = {
        "answerable": False,
        "reference_answer": "The annual revenue is not documented.",
        "expected_keywords": ["not documented"],
    }

    result = evaluate_answer_quality(
        "I cannot determine the annual revenue from the available sources.",
        reference,
    )

    assert result["answer_check"]["abstention_detected"] is True
    assert result["answer_check"]["behavior_check_passed"] is True


def test_unanswerable_case_without_abstention():
    reference = {
        "answerable": False,
        "reference_answer": "The annual revenue is not documented.",
        "expected_keywords": ["not documented"],
    }

    result = evaluate_answer_quality(
        "The annual revenue was 10 million dollars.",
        reference,
    )

    assert result["answer_check"]["abstention_detected"] is False
    assert result["answer_check"]["behavior_check_passed"] is False


def test_answerable_case_without_keywords_is_not_scored():
    reference = {
        "answerable": True,
        "reference_answer": "The system uses SQLite.",
        "expected_keywords": [],
    }

    result = evaluate_answer_quality(
        "The system uses SQLite.",
        reference,
    )

    assert result["answer_check"]["all_keywords_in_answer"] is None
    assert result["answer_check"]["behavior_check_passed"] is None
