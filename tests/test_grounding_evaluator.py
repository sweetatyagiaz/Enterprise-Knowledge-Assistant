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
