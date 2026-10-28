
"""Local rule-based RAG answer-grounding evaluator.

Run from the repository root:
    python -m tests.evaluate_grounding

This is a lexical heuristic, not semantic entailment. It makes no API calls.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPORT_DIR = Path("reports/evaluations")
REPORT_PATTERN = "live_api_evaluation_*.json"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "does",
    "for", "from", "has", "have", "how", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "uses", "using",
    "was", "were", "what", "which", "who", "with", "would", "project",
    "default", "main", "provide", "provides", "used", "configured",
    "characters", "vectors", "document", "documents",
}

TOKEN_ALIASES = {
    "dimensions": "dimension",
    "dimensional": "dimension",
    "maximum": "max",
    "overlapping": "overlap",
}

ABSTENTION_MARKERS = (
    "does not contain",
    "does not include",
    "not provided",
    "cannot determine",
    "can't determine",
    "not enough information",
    "could not find",
    "unable to determine",
    "no information about",
)


def tokenize(text: str) -> list[str]:
    """Normalize words, punctuation, and common word variants."""
    raw_tokens = re.findall(r"[a-z0-9]+", text.casefold())
    tokens = [TOKEN_ALIASES.get(token, token) for token in raw_tokens]
    return [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]


def split_claims(answer: str) -> list[str]:
    """Split answer prose and list items into claim-sized pieces."""
    claims = []

    for line in answer.splitlines():
        line = re.sub(
            r"^\s*(?:[-*•]+|\d+[.)])\s*",
            "",
            line.strip(),
        )

        # Skip headings such as "The main components are:"
        if not line or line.endswith(":"):
            continue

        parts = re.split(r"(?<=[.!?])\s+|;\s*", line)

        for part in parts:
            part = part.strip(" \t-*•")
            if part and tokenize(part):
                claims.append(part)

    return claims


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))


def best_evidence(
    source_text: str,
    claim_tokens: set[str],
) -> tuple[float, str]:
    """Find the sentence or line with the highest lexical coverage."""
    snippets = re.split(r"(?<=[.!?])\s+|\n+", source_text)

    best_score = 0.0
    best_snippet = ""

    for snippet in snippets:
        source_tokens = set(tokenize(snippet))

        if not source_tokens or not claim_tokens:
            continue

        score = len(claim_tokens & source_tokens) / len(claim_tokens)

        if score > best_score:
            best_score = score
            best_snippet = snippet.strip()

    return best_score, best_snippet

def has_specific_support(claim: str, evidence: str) -> bool:
    """Recognize a few specific, evidence-backed paraphrase patterns."""
    claim_lower = claim.casefold()
    evidence_lower = evidence.casefold()

    # Database name and exact path must both match.
    if "sqlite" in claim_lower and "sqlite" in evidence_lower:
        claim_paths = re.findall(
            r"\bdata/[a-z0-9_./-]+",
            claim_lower,
        )
        evidence_paths = re.findall(
            r"\bdata/[a-z0-9_./-]+",
            evidence_lower,
        )

        if claim_paths and any(
            path in evidence_paths for path in claim_paths
        ):
            return True

    # Recognize a file-picker instruction as upload-format evidence.
    formats = ("pdf", "docx", "txt", "markdown", "csv", "html", "xlsx")
    claim_formats = {
        fmt for fmt in formats
        if re.search(rf"\b{re.escape(fmt)}\b", claim_lower)
    }
    evidence_formats = {
        fmt for fmt in formats
        if re.search(rf"\b{re.escape(fmt)}\b", evidence_lower)
    }
    selection_instruction = re.search(
        r"\b(?:choose|select|browse for|pick)\b", evidence_lower
    )
    if (
        claim_formats
        and claim_formats <= evidence_formats
        and selection_instruction
        and re.search(r"\bfile\b", evidence_lower)
    ):
        return True

    # Every requested upload format must appear in the evidence.
    # Recognize the supported upload formats only when the claim
    # introduces no additional format absent from the evidence.
    formats = ("pdf", "docx", "txt", "markdown", "csv", "html", "xlsx")
    claim_formats = {
        fmt for fmt in formats if fmt in claim_lower
    }
    evidence_formats = {
        fmt for fmt in formats if fmt in evidence_lower
    }

    if (
        "upload" in claim_lower
        and "upload" in evidence_lower
        and {"pdf", "docx", "txt", "markdown"} <= claim_formats
        and claim_formats <= evidence_formats
    ):
        return True

    return False



NEGATION_PATTERN = re.compile(
    r"\b(?:not|never|no|doesn't|don't|isn't|aren't|"
    r"wasn't|weren't|cannot|can't|does\s+not|do\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not)\b",
    re.IGNORECASE,
)


def has_negation_conflict(claim: str, evidence: str) -> bool:
    """Flag likely polarity conflicts when claim and evidence overlap."""
    claim_negated = bool(NEGATION_PATTERN.search(claim))
    evidence_negated = bool(NEGATION_PATTERN.search(evidence))
    return claim_negated != evidence_negated



def has_model_role_conflict(claim: str, evidence: str) -> bool:
    """Detect incorrect model names assigned to documented model roles."""
    model_pattern = re.compile(
        r"\b(?:text-embedding-(?:ada-\d+|\d+(?:-\w+)*)|"
        r"gpt-\d+(?:\.\d+)?(?:-\w+)*)\b",
        re.IGNORECASE,
    )
    role_patterns = {
        "embedding": re.compile(r"\bembedding\s+model\b", re.I),
        "chat": re.compile(r"\bchat\s+model\b", re.I),
    }

    def assignments(text: str) -> dict[str, set[str]]:
        result = {}
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            models = {
                model.casefold()
                for model in model_pattern.findall(sentence)
            }
            roles = {
                role for role, pattern in role_patterns.items()
                if pattern.search(sentence)
            }
            for model in models:
                if roles:
                    result.setdefault(model, set()).update(roles)
        return result

    claim_assignments = assignments(claim)
    evidence_assignments = assignments(evidence)

    for model, claim_roles in claim_assignments.items():
        evidence_roles = evidence_assignments.get(model, set())
        if evidence_roles and claim_roles.isdisjoint(evidence_roles):
            return True

        for role in claim_roles:
            documented_models = {
                source_model
                for source_model, source_roles in evidence_assignments.items()
                if role in source_roles
            }
            if documented_models and model not in documented_models:
                return True

    return False


def has_database_path_conflict(claim: str, evidence: str) -> bool:
    """Reject a SQLite claim that specifies a different documented path."""
    if "sqlite" not in claim.casefold() or "sqlite" not in evidence.casefold():
        return False

    path_pattern = re.compile(
        r"\bdata/[a-z0-9_./-]+",
        re.IGNORECASE,
    )
    claim_paths = {
        value.rstrip(".,;:").casefold()
        for value in path_pattern.findall(claim)
    }
    evidence_paths = {
        value.rstrip(".,;:").casefold()
        for value in path_pattern.findall(evidence)
    }

    return bool(claim_paths and evidence_paths and claim_paths.isdisjoint(evidence_paths))


def has_configuration_conflict(claim: str, evidence: str) -> bool:
    """Detect numeric values assigned to the wrong named configuration key."""
    key_pattern = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
    number_pattern = re.compile(r"\b\d+(?:\.\d+)?%?\b")

    claim_keys = {key.casefold() for key in key_pattern.findall(claim)}
    claim_numbers = extract_numbers(claim)
    if not claim_keys or not claim_numbers:
        return False

    assignments = {}
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", evidence):
        keys = {key.casefold() for key in key_pattern.findall(sentence)}
        numbers = extract_numbers(sentence)
        for key in keys:
            assignments.setdefault(key, set()).update(numbers)

    for key in claim_keys:
        if key in assignments and assignments[key].isdisjoint(claim_numbers):
            return True

        # A claimed value belongs to another explicitly named setting.
        if key not in assignments:
            for other_key, values in assignments.items():
                if other_key != key and claim_numbers & values:
                    return True

    return False


def has_upload_format_conflict(claim: str, evidence: str) -> bool:
    """Reject upload-format lists that add formats absent from the source list."""
    claim_lower = claim.casefold()
    evidence_lower = evidence.casefold()

    if "upload" not in claim_lower or "upload" not in evidence_lower:
        return False

    formats = ("pdf", "docx", "txt", "markdown", "csv", "html", "xlsx")
    claim_formats = {
        fmt for fmt in formats if re.search(rf"\b{re.escape(fmt)}\b", claim_lower)
    }
    evidence_formats = {
        fmt for fmt in formats if re.search(rf"\b{re.escape(fmt)}\b", evidence_lower)
    }

    return bool(
        claim_formats
        and evidence_formats
        and not claim_formats <= evidence_formats
    )



def has_qualification_conflict(claim: str, evidence: str) -> bool:
    """Detect explicit conflicts between minimum and maximum qualifications."""
    minimum = re.compile(r"\b(?:at least|minimum of|no fewer than)\b", re.I)
    maximum = re.compile(r"\b(?:up to|at most|maximum of|no more than)\b", re.I)
    return bool(
        (minimum.search(claim) and maximum.search(evidence))
        or (maximum.search(claim) and minimum.search(evidence))
    )


def upload_only_status(claim: str, sources: list[dict]) -> str | None:
    """Assess whether an 'only' upload claim is supported or contradicted."""
    if not re.search(r"\bonly\b", claim, re.I):
        return None
    if not re.search(r"\bupload", claim, re.I):
        return None

    formats = ("pdf", "docx", "txt", "markdown", "csv", "html", "xlsx")

    def found_formats(value: str) -> set[str]:
        return {
            fmt for fmt in formats
            if re.search(rf"\b{re.escape(fmt)}\b", value, re.I)
        }

    claim_formats = found_formats(claim)
    if not claim_formats:
        return None

    source_formats = set()
    for source in sources:
        source_text = source.get("text", "")
        if isinstance(source_text, str) and re.search(r"\bupload", source_text, re.I):
            source_formats.update(found_formats(source_text))

    if source_formats - claim_formats:
        return "UNSUPPORTED"
    if claim_formats <= source_formats:
        # Mentioning a format as supported does not prove it is the only format.
        return "UNCERTAIN"
    return None


def conflicting_source_values(claim: str, sources: list[dict]) -> bool:
    """Detect multiple explicit values for a named configuration or model role."""
    key_pattern = re.compile(r"\b(CHUNK_SIZE|CHUNK_OVERLAP)\b", re.I)
    model_pattern = re.compile(
        r"\b(?:text-embedding-(?:ada-\d+|\d+(?:-\w+)*)|gpt-\d+(?:\.\d+)?(?:-\w+)*)\b",
        re.I,
    )

    claim_keys = {key.upper() for key in key_pattern.findall(claim)}
    for key in claim_keys:
        values = set()
        for source in sources:
            source_text = source.get("text", "")
            if not isinstance(source_text, str):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
                if re.search(rf"\b{key}\b", sentence, re.I):
                    values.update(extract_numbers(sentence))
        if len(values) > 1:
            return True

    roles = []
    if re.search(r"\bembedding\s+model\b", claim, re.I):
        roles.append("embedding")
    if re.search(r"\bchat\s+model\b", claim, re.I):
        roles.append("chat")

    for role in roles:
        values = set()
        for source in sources:
            source_text = source.get("text", "")
            if not isinstance(source_text, str):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
                if re.search(rf"\b{role}\s+model\b", sentence, re.I):
                    values.update(m.casefold() for m in model_pattern.findall(sentence))
        if len(values) > 1:
            return True

    return False


def has_database_engine_conflict(claim: str, evidence: str) -> bool:
    """Detect a database-engine mismatch in the relevant evidence snippet."""
    engines = ("sqlite", "postgresql", "postgres", "mysql", "mariadb", "mongodb")

    def normalize_engines(text: str) -> set[str]:
        found = {
            engine
            for engine in engines
            if re.search(rf"\b{engine}\b", text, re.I)
        }
        if "postgres" in found:
            found.add("postgresql")
        return found

    claim_engines = normalize_engines(claim)
    evidence_engines = normalize_engines(evidence)

    return bool(
        claim_engines
        and evidence_engines
        and claim_engines.isdisjoint(evidence_engines)
    )


def has_unsupported_database_model_link(
    claim: str, sources: list[dict]
) -> bool:
    """Avoid inferring a database-to-model relationship from separate facts."""
    model_pattern = re.compile(
        r"\b(?:text-embedding-(?:ada-\d+|\d+(?:-\w+)*)|gpt-\d+(?:\.\d+)?(?:-\w+)*)\b",
        re.I,
    )
    engines = ("sqlite", "postgresql", "postgres", "mysql", "mariadb", "mongodb")

    claim_has_engine = any(re.search(rf"\b{engine}\b", claim, re.I) for engine in engines)
    claim_has_model = bool(model_pattern.search(claim))
    relationship_words = re.search(
        r"\b(?:uses|using|stores|store|stores\s+.*vectors|to\s+store)\b",
        claim,
        re.I,
    )
    if not (claim_has_engine and claim_has_model and relationship_words):
        return False

    for source in sources:
        source_text = source.get("text", "")
        if not isinstance(source_text, str):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", source_text):
            has_engine = any(
                re.search(rf"\b{engine}\b", sentence, re.I) for engine in engines
            )
            has_model = bool(model_pattern.search(sentence))
            if has_engine and has_model:
                return False

    return True

def has_storage_paraphrase_support(claim: str, evidence: str) -> bool:
    """Recognize a narrow storage paraphrase when the database matches."""
    claim_lower = claim.casefold()
    evidence_lower = evidence.casefold()

    database_engines = (
        "sqlite", "postgresql", "postgres", "mysql", "mariadb", "mongodb",
    )

    shared_engines = [
        engine
        for engine in database_engines
        if re.search(rf"\b{engine}\b", claim_lower)
        and re.search(rf"\b{engine}\b", evidence_lower)
    ]
    if not shared_engines:
        return False

    claim_storage = re.search(
        r"\b(?:persist(?:ed|s|ing)?|store(?:d|s|ing)?|save(?:d|s|ing)?)\b",
        claim_lower,
    )
    evidence_storage = re.search(
        r"\b(?:persist(?:ed|s|ing)?|store(?:d|s|ing)?|save(?:d|s|ing)?)\b",
        evidence_lower,
    )

    claim_content = re.search(
        r"\b(?:chunks?|segments?|passages?|pieces?)\b", claim_lower
    )
    evidence_content = re.search(
        r"\b(?:chunks?|segments?|passages?|pieces?)\b", evidence_lower
    )

    return bool(
        claim_storage
        and evidence_storage
        and claim_content
        and evidence_content
    )

def evaluate_claim(claim: str, sources: list[dict]) -> dict:
    claim_tokens = set(tokenize(claim))

    if not sources:
        return {
            "claim": claim,
            "label": "UNCERTAIN",
            "lexical_coverage": 0.0,
            "evidence": None,
            "reason": "No retrieved sources were returned.",
        }

    best = {
        "score": 0.0,
        "filename": None,
        "text": "",
        "full_text": "",
    }

    for source in sources:
        source_text = source.get("text", "")

        if not isinstance(source_text, str):
            continue

        score, snippet = best_evidence(source_text, claim_tokens)

        if score > best["score"]:
            best = {
                "score": score,
                "filename": source.get("filename", "Unknown source"),
                "text": snippet,
                "full_text": source_text,
            }

    # Check numbers against the best-matching evidence, not all sources.
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(best["text"])
    missing_numbers = sorted(claim_numbers - evidence_numbers)

    full_evidence = best["full_text"] or best["text"]
    all_evidence = "\n".join(
        source["text"]
        for source in sources
        if isinstance(source.get("text"), str)
    )

    if conflicting_source_values(claim, sources):
        label = "UNCERTAIN"
        reason = "Multiple conflicting values are documented across the retrieved sources."
    elif upload_only_status(claim, sources) == "UNSUPPORTED":
        label = "UNSUPPORTED"
        reason = "The evidence lists additional upload formats beyond those claimed as exclusive."
    elif upload_only_status(claim, sources) == "UNCERTAIN":
        label = "UNCERTAIN"
        reason = "The evidence supports the named format but does not establish exclusivity."
    elif has_qualification_conflict(claim, all_evidence):
        label = "UNSUPPORTED"
        reason = "The claim conflicts with the documented minimum or maximum qualification."
    elif has_database_engine_conflict(claim, best["text"]):
        label = "UNSUPPORTED"
        reason = "The claimed database engine conflicts with the documented engine."
    elif has_unsupported_database_model_link(claim, sources):
        label = "UNSUPPORTED"
        reason = "The claimed relationship between the database and model is not established by one source statement."
    elif has_model_role_conflict(claim, all_evidence):
        label = "UNSUPPORTED"
        reason = "The model name conflicts with the documented model-role assignment."
    elif has_database_path_conflict(claim, full_evidence):
        label = "UNSUPPORTED"
        reason = "The claimed SQLite database path conflicts with the documented path."
    elif has_configuration_conflict(claim, full_evidence):
        label = "UNSUPPORTED"
        reason = "A configuration value is attributed to the wrong setting."
    elif has_upload_format_conflict(claim, full_evidence):
        label = "UNSUPPORTED"
        reason = "The claim includes an upload format absent from the documented format list."
    elif best["score"] >= 0.35 and has_negation_conflict(
        claim, best["text"]
    ):
        label = "UNSUPPORTED"
        reason = (
            "Claim and best-matching evidence appear to conflict in negation."
        )
    elif missing_numbers:
        label = "UNSUPPORTED"
        reason = (
            "Claim number(s) are absent from the best-matching evidence snippet."
        )
    elif (has_specific_support(claim, best["text"]) or has_storage_paraphrase_support(claim, best["text"])):
        label = "SUPPORTED"
        reason = (
            "Claim matches a recognized, specific fact in the evidence snippet."
        )
    elif best["score"] >= 0.70:
        label = "SUPPORTED"
        reason = (
            "High normalized lexical coverage in one retrieved source snippet."
        )
    elif best["score"] >= 0.35:
        label = "UNCERTAIN"
        reason = "Partial lexical overlap; review the evidence manually."
    else:
        label = "UNSUPPORTED"
        reason = "Insufficient lexical overlap with retrieved source text."

    return {
        "claim": claim,
        "label": label,
        "lexical_coverage": round(best["score"], 3),
        "evidence": (
            {
                "filename": best["filename"],
                "snippet": best["text"],
            }
            if best["text"]
            else None
        ),
        "missing_numbers": missing_numbers,
        "reason": reason,
    }


def evaluate_result(result: dict) -> dict:
    answer = result.get("answer") or ""
    sources = result.get("sources") or []

    if any(marker in answer.casefold() for marker in ABSTENTION_MARKERS):
        claims = [
            {
                "claim": answer.strip(),
                "label": "ABSTENTION",
                "lexical_coverage": None,
                "evidence": None,
                "reason": (
                    "Recognized abstention phrase; this does not prove "
                    "the whole knowledge base lacks the answer."
                ),
            }
        ]
    else:
        claims = [
            evaluate_claim(claim, sources)
            for claim in split_claims(answer)
        ]

    scored = [
        claim for claim in claims
        if claim["label"] in {
            "SUPPORTED", "UNSUPPORTED", "UNCERTAIN"
        }
    ]

    supported = sum(
        claim["label"] == "SUPPORTED" for claim in scored
    )
    unsupported = sum(
        claim["label"] == "UNSUPPORTED" for claim in scored
    )
    uncertain = sum(
        claim["label"] == "UNCERTAIN" for claim in scored
    )

    return {
        "id": result.get("id"),
        "question": result.get("question"),
        "original_status": result.get("status"),
        "claims": claims,
        "summary": {
            "claim_count": len(scored),
            "supported": supported,
            "unsupported": unsupported,
            "uncertain": uncertain,
            "supported_rate": (
                round(supported / len(scored), 3)
                if scored else None
            ),
        },
    }


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        REPORT_DIR.glob(REPORT_PATTERN),
        key=lambda path: path.stat().st_mtime,
    )

    if not candidates:
        print(f"No live API evaluation reports found in {REPORT_DIR}.")
        print("Run python -m tests.evaluate_live_api first.")
        return 1

    input_path = candidates[-1]
    source_report = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    results = [
        evaluate_result(item)
        for item in source_report.get("results", [])
    ]

    claims = [
        claim
        for result in results
        for claim in result["claims"]
        if claim["label"] in {
            "SUPPORTED", "UNSUPPORTED", "UNCERTAIN"
        }
    ]

    counts = {
        label.lower(): sum(
            claim["label"] == label for claim in claims
        )
        for label in ("SUPPORTED", "UNSUPPORTED", "UNCERTAIN")
    }

    summary = {
        "input_report": str(input_path),
        "questions_evaluated": len(results),
        "claims_evaluated": len(claims),
        **counts,
        "supported_rate": (
            round(counts["supported"] / len(claims), 3)
            if claims else None
        ),
    }

    timestamp = datetime.now(timezone.utc)

    report = {
        "timestamp_utc": timestamp.isoformat(),
        "method": "local lexical overlap heuristic; no LLM/API calls",
        "limitations": [
            "Lexical overlap is not semantic entailment or proof of factual correctness.",
            "Correct paraphrases may be labelled uncertain or unsupported.",
            "Matching text may omit context, qualifications, or contradictions.",
            "A number appearing in a matching snippet does not prove fact linkage.",
            "Abstention does not prove the full knowledge base lacks an answer.",
            "Review evidence snippets manually before treating labels as ground truth.",
        ],
        "summary": summary,
        "results": results,
    }

    output_path = REPORT_DIR / (
        f"grounding_evaluation_"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Local Rule-Based Grounding Evaluation ===")
    print(f"Input report: {input_path}")
    print(f"Questions evaluated: {summary['questions_evaluated']}")
    print(f"Claims evaluated: {summary['claims_evaluated']}")
    print(f"Supported: {counts['supported']}")
    print(f"Unsupported: {counts['unsupported']}")
    print(f"Uncertain: {counts['uncertain']}")
    print(f"Supported rate (heuristic): {summary['supported_rate']}")
    print(f"JSON report saved: {output_path}")
    print("Reminder: labels are lexical heuristics, not semantic verification.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
