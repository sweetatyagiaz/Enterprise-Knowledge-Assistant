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
