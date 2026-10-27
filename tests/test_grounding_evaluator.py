from ai_core.retrieval.grounding_evaluator import evaluate_grounding


def test_unlisted_upload_format_uses_deterministic_policy():
    def unexpected_llm_call():
        raise AssertionError("LLM judge should not be called")

    result = evaluate_grounding(
        claim="Users can upload executable files.",
        evidence=(
            "Supported upload formats are PDF, DOCX, TXT, and Markdown."
        ),
        llm_judge=unexpected_llm_call,
    )

    assert result["verdict"] == "NOT_ENOUGH_EVIDENCE"
    assert result["verdict_source"] == "deterministic_policy"


def test_explicitly_prohibited_format_is_contradicted():
    def unexpected_llm_call():
        raise AssertionError("LLM judge should not be called")

    result = evaluate_grounding(
        claim="Users can upload executable files.",
        evidence=(
            "Supported upload formats are PDF, DOCX, TXT, and Markdown. "
            "Executable files are explicitly prohibited."
        ),
        llm_judge=unexpected_llm_call,
    )

    assert result["verdict"] == "CONTRADICTED"
    assert result["verdict_source"] == "deterministic_policy"


def test_unrelated_claim_uses_llm_judge():
    expected_judgment = {
        "verdict": "SUPPORTED",
        "rationale": "Evidence supports the claim.",
        "unsupported_claims": [],
    }

    def mock_llm_judge():
        return expected_judgment

    result = evaluate_grounding(
        claim="The configured embedding dimension is 1536.",
        evidence="Embedding dimensions are configured as 1536.",
        llm_judge=mock_llm_judge,
    )

    assert result["verdict"] == "SUPPORTED"
    assert result["verdict_source"] == "llm_judge"
    assert result["rationale"] == "Evidence supports the claim."

def test_empty_evidence_uses_llm_judge():
    result = evaluate_grounding(
        claim="The embedding model uses 1536 dimensions.",
        evidence="",
        llm_judge=lambda: {
            "verdict": "NOT_ENOUGH_EVIDENCE",
            "rationale": "No evidence was provided.",
            "unsupported_claims": ["The embedding model uses 1536 dimensions."],
        },
    )

    assert result["verdict"] == "NOT_ENOUGH_EVIDENCE"
    assert result["verdict_source"] == "llm_judge"


def test_llm_judge_result_fields_are_preserved():
    judgment = {
        "verdict": "SUPPORTED",
        "rationale": "The evidence directly supports the claim.",
        "unsupported_claims": [],
        "confidence": 0.95,
    }

    result = evaluate_grounding(
        claim="The database is SQLite.",
        evidence="The project stores data in SQLite.",
        llm_judge=lambda: judgment,
    )

    assert result["confidence"] == 0.95
    assert result["rationale"] == judgment["rationale"]
    assert result["verdict_source"] == "llm_judge"


def test_llm_judge_exception_propagates():
    def failing_judge():
        raise RuntimeError("Judge unavailable")

    try:
        evaluate_grounding(
            claim="The database is SQLite.",
            evidence="The project uses SQLite.",
            llm_judge=failing_judge,
        )
    except RuntimeError as exc:
        assert str(exc) == "Judge unavailable"
    else:
        raise AssertionError("Expected RuntimeError")


def test_deterministic_policy_does_not_call_llm_judge():
    def unexpected_judge():
        raise AssertionError("LLM judge should not be called")

    result = evaluate_grounding(
        claim="Users can upload executable files.",
        evidence="Supported upload formats are PDF, DOCX, TXT, and Markdown.",
        llm_judge=unexpected_judge,
    )

    assert result["verdict_source"] == "deterministic_policy"


