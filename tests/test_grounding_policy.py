from ai_core.retrieval.grounding_policy import (
    classify_unsupported_option,
    explicitly_prohibited,
    list_is_explicitly_exhaustive,
)


def test_unlisted_format_is_not_automatically_contradicted():
    evidence = (
        "Supported upload formats are PDF, DOCX, TXT, and Markdown."
    )

    assert classify_unsupported_option(
        "Users can upload executable files.",
        evidence,
    ) == "NOT_ENOUGH_EVIDENCE"


def test_explicit_prohibition_is_contradicted():
    evidence = (
        "Supported upload formats are PDF, DOCX, TXT, and Markdown. "
        "Executable files are explicitly prohibited."
    )

    assert explicitly_prohibited(
        "Users can upload executable files.",
        evidence,
    )


def test_explicitly_exhaustive_list_is_detected():
    evidence = (
        "The only supported upload formats are PDF, DOCX, TXT, "
        "and Markdown."
    )

    assert list_is_explicitly_exhaustive(evidence)
    assert classify_unsupported_option(
        "Users can upload executable files.",
        evidence,
    ) == "CONTRADICTED"


def test_unrelated_prohibition_does_not_match_claim_subject():
    evidence = "Executable files are prohibited."
    claim = "Users can upload Markdown files."

    assert not explicitly_prohibited(claim, evidence)

def test_unrelated_evidence_does_not_classify_upload_claim():
    result = classify_unsupported_option(
        "Users can upload executable files.",
        "The application uses FastAPI and SQLite.",
    )

    assert result is None


def test_upload_policy_is_case_insensitive():
    result = classify_unsupported_option(
        "Users can upload EXECUTABLE files.",
        "Supported upload formats are PDF, DOCX, TXT, and Markdown.",
    )

    assert result == "NOT_ENOUGH_EVIDENCE"


def test_non_exhaustive_list_does_not_imply_contradiction():
    evidence = "Supported upload formats include PDF, DOCX, TXT, and Markdown."

    assert not list_is_explicitly_exhaustive(evidence)
    assert classify_unsupported_option(
        "Users can upload executable files.",
        evidence,
    ) == "NOT_ENOUGH_EVIDENCE"


def test_unrelated_claim_is_outside_upload_format_policy():
    result = classify_unsupported_option(
        "The database supports SQL queries.",
        "Supported upload formats are PDF, DOCX, TXT, and Markdown.",
    )

    assert result is None


