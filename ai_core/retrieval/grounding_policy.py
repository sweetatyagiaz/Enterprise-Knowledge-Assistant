"""Deterministic policy checks for semantic grounding evaluation."""

import re


def explicitly_prohibited(claim: str, evidence: str) -> bool:
    """
    Detect a claim that conflicts with an explicit prohibition in evidence.

    This is a narrow lexical check, not a general natural-language entailment
    engine. Unrecognized wording should be left to the LLM judge.
    """
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()

    prohibition_patterns = [
        r"\b(?:is|are|was|were)\s+(?:explicitly\s+)?prohibited\b",
        r"\bnot\s+allowed\b",
        r"\bnot\s+supported\b",
        r"\bforbidden\b",
        r"\bmust\s+not\b",
        r"\bdo\s+not\s+allow\b",
    ]

    if not any(
        re.search(pattern, evidence_lower)
        for pattern in prohibition_patterns
    ):
        return False

    # Require the evidence and claim to share a meaningful subject term.
    # This avoids treating an unrelated prohibition as a contradiction.
    ignored = {
        "the", "and", "for", "can", "users", "upload", "files", "file",
        "are", "is", "not", "allowed", "prohibited", "explicitly",
    }

    claim_terms = {
        word for word in re.findall(r"[a-z0-9][a-z0-9._-]*", claim_lower)
        if len(word) > 2 and word not in ignored
    }

    evidence_terms = {
        word for word in re.findall(r"[a-z0-9][a-z0-9._-]*", evidence_lower)
        if len(word) > 2 and word not in ignored
    }

    return bool(claim_terms & evidence_terms)


def list_is_explicitly_exhaustive(evidence: str) -> bool:
    """Check whether evidence explicitly says a list is exhaustive."""
    patterns = [
        r"\bthe only\b",
        r"\bonly the following\b",
        r"\bthese are the only\b",
        r"\bno other\b",
        r"\bexclusively\b",
    ]

    evidence_lower = evidence.lower()
    return any(
        re.search(pattern, evidence_lower)
        for pattern in patterns
    )


def classify_unsupported_option(
    claim: str,
    evidence: str,
) -> str | None:
    """
    Apply a narrow upload-format policy.

    Return None when the case is outside this policy's scope so the LLM
    judge can handle it. This is not a general entailment engine.
    """
    claim_lower = claim.lower()
    evidence_lower = evidence.lower()

    # Restrict this rule to the upload-format scenario it was designed for.
    if "supported upload formats" not in evidence_lower:
        return None

    if "executable" not in claim_lower:
        return None

    if explicitly_prohibited(claim, evidence):
        return "CONTRADICTED"

    if list_is_explicitly_exhaustive(evidence):
        return "CONTRADICTED"

    # The option is unlisted, but the evidence does not explicitly prohibit it
    # or declare the supported-format list exhaustive.
    return "NOT_ENOUGH_EVIDENCE"
