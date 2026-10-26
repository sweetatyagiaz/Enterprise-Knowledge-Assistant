from ai_core.retrieval.grounding_policy import (
    classify_unsupported_option,
    explicitly_prohibited,
    list_is_explicitly_exhaustive,
)


def test_unlisted_executable_without_exhaustive_list_is_uncertain():
    evidence = (
        "Supported upload formats are PDF, DOCX, TXT, and Markdown."
    )
    claim = "Users can upload executable files."

    assert classify_unsupported_option(claim, evidence) == (
        "NOT_ENOUGH_EVIDENCE"
    )


def test_explicitly_prohibited_executable_is_contradicted():
    evidence = (
        "Supported upload formats are PDF, DOCX, TXT, and Markdown. "
        "Executable files are explicitly prohibited."
    )
    claim = "Users can upload executable files."

    assert classify_unsupported_option(claim, evidence) == "CONTRADICTED"


def test_non_upload_claim_does_not_activate_upload_policy():
    evidence = "The project uses SQLite to store document chunks."
    claim = "Executable files are supported."

    assert classify_unsupported_option(claim, evidence) is None


def test_exhaustive_wording_is_detected():
    evidence = "The only supported upload formats are PDF and DOCX."

    assert list_is_explicitly_exhaustive(evidence)


def test_unrelated_prohibition_does_not_match_unrelated_claim():
    evidence = "Executable files are prohibited."
    claim = "Users can upload video files."

    assert not explicitly_prohibited(claim, evidence)
