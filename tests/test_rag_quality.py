from tests.evaluate_rag_quality import evaluate_case, rate


def make_result(answer, sources=None, case_id="RAG-TEST"):
    return {
        "id": case_id,
        "question": "What is the default embedding model?",
        "answer": answer,
        "status": "PASS",
        "sources": sources or [],
    }


def make_source(text, filename="README.md", score=0.8):
    return {
        "filename": filename,
        "text": text,
        "score": score,
        "hybrid_score": score,
    }


def make_reference(answerable=True, keywords=None):
    return {
        "id": "RAG-TEST",
        "answerable": answerable,
        "expected_source": "README.md" if answerable else None,
        "expected_keywords": keywords or [],
    }


def test_correct_abstention_passes_for_unanswerable_question():
    result = make_result(
        "The provided document context does not contain information "
        "about annual revenue."
    )
    reference = make_reference(answerable=False)

    evaluated = evaluate_case(result, reference)

    assert evaluated["answer_check"]["abstention_detected"] is True
    assert evaluated["answer_check"]["behavior_check_passed"] is True


def test_answerable_question_fails_when_expected_keyword_is_missing():
    result = make_result(
        "The project uses a different embedding model.",
        [make_source("The default embedding model is text-embedding-3-small.")],
    )
    reference = make_reference(
        keywords=["text-embedding-3-small"]
    )

    evaluated = evaluate_case(result, reference)

    assert evaluated["answer_check"]["all_keywords_in_answer"] is False
    assert evaluated["answer_check"]["behavior_check_passed"] is False


def test_empty_retrieval_has_zero_sources_and_no_keyword_coverage():
    result = make_result("I could not find relevant information.")
    reference = make_reference(
        keywords=["text-embedding-3-small"]
    )

    evaluated = evaluate_case(result, reference)

    assert evaluated["retrieval"]["source_count"] == 0
    assert evaluated["retrieval"]["all_keywords_in_top1"] is False
    assert evaluated["retrieval"]["all_keywords_in_topk"] is False
    assert evaluated["retrieval"]["top_ranked_score"] is None


def test_keyword_coverage_can_be_distributed_across_sources():
    result = make_result(
        "The default embedding model is text-embedding-3-small.",
        [
            make_source("The default embedding model is text-embedding."),
            make_source("The configured model name is text-embedding-3-small."),
        ],
    )
    reference = make_reference(
        keywords=["default embedding model", "text-embedding-3-small"]
    )

    evaluated = evaluate_case(result, reference)

    assert evaluated["retrieval"]["all_keywords_in_top1"] is False
    assert evaluated["retrieval"]["all_keywords_in_topk"] is True


def test_grounding_evaluation_is_included_in_combined_result():
    result = make_result(
        "The default embedding model is text-embedding-3-small.",
        [
            make_source(
                "The default embedding model is text-embedding-3-small."
            )
        ],
    )
    reference = make_reference(
        keywords=["text-embedding-3-small"]
    )

    evaluated = evaluate_case(result, reference)

    assert "grounding" in evaluated
    assert "summary" in evaluated["grounding"]
    assert evaluated["grounding"]["summary"]["claim_count"] >= 1


def test_rate_handles_zero_denominator():
    assert rate(0, 0) is None


def test_rate_calculates_ratio():
    assert rate(3, 4) == 0.75


def test_reference_answer_overlap_is_one_for_matching_answers():
    from tests.evaluate_rag_quality import reference_answer_overlap

    result = reference_answer_overlap(
        "SQLite stores document chunks.",
        "SQLite stores document chunks.",
    )

    assert result["reference_available"] is True
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_reference_answer_overlap_handles_partial_match():
    from tests.evaluate_rag_quality import reference_answer_overlap

    result = reference_answer_overlap(
        "The project uses SQLite storage.",
        "The project uses SQLite to store document chunks.",
    )

    assert result["reference_available"] is True
    assert result["f1"] is not None
    assert 0.0 < result["f1"] < 1.0


def test_reference_answer_overlap_handles_missing_reference():
    from tests.evaluate_rag_quality import reference_answer_overlap

    result = reference_answer_overlap("Some answer.", None)

    assert result == {
        "reference_available": False,
        "precision": None,
        "recall": None,
        "f1": None,
    }


def test_reference_answer_overlap_ignores_case_and_punctuation():
    from tests.evaluate_rag_quality import reference_answer_overlap

    result = reference_answer_overlap(
        "SQLite, VECTOR STORE!",
        "sqlite vector store",
    )

    assert result["f1"] == 1.0
