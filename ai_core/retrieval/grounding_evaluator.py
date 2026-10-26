"""Routing between deterministic grounding rules and an LLM judge."""


def evaluate_grounding(
    *,
    claim: str,
    evidence: str,
    llm_judge,
) -> dict:
    """
    Apply a deterministic policy when it recognizes a case.
    Otherwise, delegate to the supplied LLM judge callable.

    llm_judge must be a zero-argument callable returning a judgment dict.
    """
    from ai_core.retrieval.grounding_policy import (
        classify_unsupported_option,
    )

    policy_verdict = classify_unsupported_option(
        claim=claim,
        evidence=evidence,
    )

    if policy_verdict is not None:
        return {
            "verdict": policy_verdict,
            "rationale": (
                "Deterministic grounding policy applied to "
                "the upload-format claim."
            ),
            "unsupported_claims": (
                [claim]
                if policy_verdict == "NOT_ENOUGH_EVIDENCE"
                else []
            ),
            "verdict_source": "deterministic_policy",
        }

    judgment = llm_judge()
    return {
        **judgment,
        "verdict_source": "llm_judge",
    }
