"""Combined retrieval, grounding, and answerability evaluation.

Run from the repository root:
    python -m tests.evaluate_rag_quality

Uses the latest live API evaluation report. Does not call the API.
Keyword checks are heuristic proxies, not semantic correctness judgments.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tests.evaluate_grounding import evaluate_result


REPORT_DIR = Path("reports/evaluations")
REPORT_PATTERN = "live_api_evaluation_*.json"
DATASET_PATH = Path("tests/data/rag_ground_truth.json")
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


def contains_keyword(text: str, keyword: str) -> bool:
    return keyword.casefold() in text.casefold()


def contains_all_keywords(text: str, keywords: list[str]) -> bool:
    return all(contains_keyword(text, keyword) for keyword in keywords)


def detect_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def safe_score(source: dict) -> float | None:
    value = source.get("hybrid_score", source.get("score"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def reference_answer_overlap(answer: str, reference_answer: str | None) -> dict:
    """Calculate lexical token overlap; this is not semantic evaluation."""
    if not reference_answer:
        return {
            "reference_available": False,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for",
        "from", "has", "have", "in", "is", "it", "of", "on", "or",
        "that", "the", "their", "this", "to", "was", "with",
    }

    def tokens(text: str) -> set[str]:
        return {
            token for token in re.findall(r"[a-z0-9]+", text.casefold())
            if token not in stopwords
        }

    answer_tokens = tokens(answer)
    reference_tokens = tokens(reference_answer)

    if not answer_tokens or not reference_tokens:
        return {
            "reference_available": True,
            "precision": None,
            "recall": None,
            "f1": None,
        }

    overlap = len(answer_tokens & reference_tokens)
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "reference_available": True,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def evaluate_case(result: dict, reference: dict) -> dict:
    answer = str(result.get("answer") or "")
    sources = result.get("sources") or []
    keywords = reference.get("expected_keywords") or []
    answerable = bool(reference["answerable"])
    expected_source = reference.get("expected_source")

    top_text = str(sources[0].get("text", "")) if sources else ""
    all_source_text = "\n".join(
        str(source.get("text", ""))
        for source in sources
    )

    top1_keyword_coverage = (
        sum(contains_keyword(top_text, keyword) for keyword in keywords)
        / len(keywords)
        if keywords else None
    )
    topk_keyword_coverage = (
        sum(
            contains_keyword(all_source_text, keyword)
            for keyword in keywords
        ) / len(keywords)
        if keywords else None
    )
    answer_keyword_coverage = (
        sum(contains_keyword(answer, keyword) for keyword in keywords)
        / len(keywords)
        if keywords else None
    )

    top1_all_keywords = (
        contains_all_keywords(top_text, keywords)
        if keywords else None
    )
    topk_all_keywords = (
        contains_all_keywords(all_source_text, keywords)
        if keywords else None
    )
    answer_all_keywords = (
        contains_all_keywords(answer, keywords)
        if keywords else None
    )

    source_hit = (
        any(
            source.get("filename") == expected_source
            for source in sources
        )
        if expected_source else None
    )

    abstention_detected = detect_abstention(answer)
    reference_comparison = reference_answer_overlap(
        answer,
        reference.get("reference_answer"),
    )

    if answerable:
        behavior_correct = answer_all_keywords is True
        behavior_reason = (
            "All expected keywords appear in the answer."
            if behavior_correct
            else "One or more expected keywords are absent from the answer."
        )
    else:
        behavior_correct = abstention_detected
        behavior_reason = (
            "Recognized an abstention phrase for an unanswerable question."
            if behavior_correct
            else "No recognized abstention phrase for an unanswerable question."
        )

    grounding = evaluate_result(result)
    claim_summary = grounding["summary"]

    scores = [
        score for score in (safe_score(source) for source in sources)
        if score is not None
    ]

    return {
        "id": result.get("id"),
        "question": result.get("question"),
        "answerable": answerable,
        "status": result.get("status"),
        "answer": answer,
        "retrieval": {
            "source_count": len(sources),
            "expected_source": expected_source,
            "expected_source_found": source_hit,
            "top1_keyword_coverage": (
                round(top1_keyword_coverage, 3)
                if top1_keyword_coverage is not None else None
            ),
            "topk_keyword_coverage": (
                round(topk_keyword_coverage, 3)
                if topk_keyword_coverage is not None else None
            ),
            "all_keywords_in_top1": top1_all_keywords,
            "all_keywords_in_topk": topk_all_keywords,
            "ranked_scores": scores,
            "top_ranked_score": scores[0] if scores else None,
        },
        "reference_answer_comparison": reference_comparison,
        "answer_check": {
            "expected_keywords": keywords,
            "answer_keyword_coverage": (
                round(answer_keyword_coverage, 3)
                if answer_keyword_coverage is not None else None
            ),
            "all_keywords_in_answer": answer_all_keywords,
            "abstention_detected": abstention_detected,
            "behavior_check_passed": behavior_correct,
            "behavior_check_reason": behavior_reason,
        },
        "grounding": grounding,
    }


def rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 3) if denominator else None


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

    if not DATASET_PATH.exists():
        print(f"Evaluation dataset not found: {DATASET_PATH}")
        return 1

    input_path = candidates[-1]
    source_report = json.loads(input_path.read_text(encoding="utf-8"))
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    references = {item["id"]: item for item in dataset}
    results = []
    missing_references = []

    for result in source_report.get("results", []):
        case_id = result.get("id")
        reference = references.get(case_id)

        if reference is None:
            missing_references.append(case_id)
            continue

        results.append(evaluate_case(result, reference))

    answerable_cases = [item for item in results if item["answerable"]]
    unanswerable_cases = [item for item in results if not item["answerable"]]

    answerable_behavior_passes = sum(
        item["answer_check"]["behavior_check_passed"]
        for item in answerable_cases
    )
    abstention_passes = sum(
        item["answer_check"]["behavior_check_passed"]
        for item in unanswerable_cases
    )

    top1_hits = sum(
        item["retrieval"]["all_keywords_in_top1"] is True
        for item in answerable_cases
    )
    topk_hits = sum(
        item["retrieval"]["all_keywords_in_topk"] is True
        for item in answerable_cases
    )
    source_hits = sum(
        item["retrieval"]["expected_source_found"] is True
        for item in answerable_cases
    )

    reference_scores = [
        item["reference_answer_comparison"]["f1"]
        for item in answerable_cases
        if item["reference_answer_comparison"]["f1"] is not None
    ]

    all_claims = [
        claim
        for item in results
        for claim in item["grounding"]["claims"]
        if claim["label"] in {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}
    ]
    grounding_counts = {
        label.lower(): sum(claim["label"] == label for claim in all_claims)
        for label in ("SUPPORTED", "UNSUPPORTED", "UNCERTAIN")
    }

    summary = {
        "cases_evaluated": len(results),
        "cases_missing_reference": missing_references,
        "answerable_cases": len(answerable_cases),
        "unanswerable_cases": len(unanswerable_cases),
        "retrieval": {
            "expected_source_hit_rate": rate(
                source_hits, len(answerable_cases)
            ),
            "all_keywords_in_top1_rate": rate(
                top1_hits, len(answerable_cases)
            ),
            "all_keywords_in_topk_rate": rate(
                topk_hits, len(answerable_cases)
            ),
        },
        "answer_checks": {
            "answerable_keyword_check_pass_rate": rate(
                answerable_behavior_passes, len(answerable_cases)
            ),
            "unanswerable_abstention_rate": rate(
                abstention_passes, len(unanswerable_cases)
            ),
        },
        "reference_answer_comparison": {
            "answerable_cases_with_reference_score": len(reference_scores),
            "mean_lexical_f1": (
                round(sum(reference_scores) / len(reference_scores), 3)
                if reference_scores else None
            ),
            "interpretation": (
                "Lexical token overlap only; not a semantic correctness score."
            ),
        },
        "grounding_claims": {
            "claim_count": len(all_claims),
            **grounding_counts,
            "supported_rate": rate(
                grounding_counts["supported"], len(all_claims)
            ),
        },
    }

    timestamp = datetime.now(timezone.utc)
    report = {
        "timestamp_utc": timestamp.isoformat(),
        "input_report": str(input_path),
        "reference_dataset": str(DATASET_PATH),
        "method": (
            "Keyword-based retrieval and answer checks combined with the "
            "existing local lexical grounding evaluator."
        ),
        "limitations": [
            "Keyword coverage is not semantic relevance or correctness.",
            "Expected-source hits do not prove a retrieved passage is relevant.",
            "Retrieval scores are diagnostics, not calibrated probabilities.",
            "Abstention markers do not prove that abstention was warranted.",
            "Grounding labels use a local lexical heuristic, not semantic entailment.",
            "Answer keyword checks may miss correct paraphrases.",
        ],
        "summary": summary,
        "results": results,
    }

    output_path = REPORT_DIR / (
        f"rag_quality_evaluation_"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Combined RAG Quality Evaluation ===")
    print(f"Input report: {input_path}")
    print(f"Cases evaluated: {summary['cases_evaluated']}")
    print(f"Answerable cases: {summary['answerable_cases']}")
    print(f"Unanswerable cases: {summary['unanswerable_cases']}")
    print(
        "All keywords in top-1 retrieval (answerable): "
        f"{summary['retrieval']['all_keywords_in_top1_rate']}"
    )
    print(
        "All keywords in top-K retrieval (answerable): "
        f"{summary['retrieval']['all_keywords_in_topk_rate']}"
    )
    print(
        "Answer keyword check pass rate: "
        f"{summary['answer_checks']['answerable_keyword_check_pass_rate']}"
    )
    print(
        "Abstention rate on unanswerable questions: "
        f"{summary['answer_checks']['unanswerable_abstention_rate']}"
    )
    print(
        "Mean reference-answer lexical F1 (answerable cases): "
        f"{summary['reference_answer_comparison']['mean_lexical_f1']}"
    )
    print(
        "Cases with reference-answer scores: "
        f"{summary['reference_answer_comparison']['answerable_cases_with_reference_score']}"
    )
    print(f"Grounding claims evaluated: {summary['grounding_claims']['claim_count']}")
    print(f"Grounding supported rate: {summary['grounding_claims']['supported_rate']}")
    print(f"JSON report saved: {output_path}")
    print("Reminder: these metrics are heuristic diagnostics, not proof of correctness.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
