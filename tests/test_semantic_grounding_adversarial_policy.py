def classify_claim_against_evidence(
    claim: str,
    evidence: str,
    *,
    evidence_explicitly_exhaustive: bool = False,
    evidence_explicitly_prohibits: bool = False,
) -> str:
    """
    Minimal policy helper for testing the distinction between missing
    evidence and explicit contradiction. This is not the production LLM judge.
    """
    if evidence_explicitly_prohibits:
        return "CONTRADICTED"

    if evidence_explicitly_exhaustive:
        return "CONTRADICTED"

    return "NOT_ENOUGH_EVIDENCE"


def test_unlisted_format_is_not_automatically_contradicted():
    verdict = classify_claim_against_evidence(
        claim="Users can upload executable files.",
        evidence=(
            "Supported upload formats are PDF, DOCX, TXT, and Markdown."
        ),
    )

    assert verdict == "NOT_ENOUGH_EVIDENCE"


def test_explicitly_prohibited_format_is_contradicted():
    verdict = classify_claim_against_evidence(
        claim="Users can upload executable files.",
        evidence=(
            "Supported upload formats are PDF, DOCX, TXT, and Markdown. "
            "Executable files are explicitly prohibited."
        ),
        evidence_explicitly_prohibits=True,
    )

    assert verdict == "CONTRADICTED"


def test_unlisted_format_in_an_explicitly_exhaustive_list_is_contradicted():
    verdict = classify_claim_against_evidence(
        claim="Users can upload executable files.",
        evidence=(
            "The only supported upload formats are PDF, DOCX, TXT, "
            "and Markdown."
        ),
        evidence_explicitly_exhaustive=True,
    )

    assert verdict == "CONTRADICTED"
