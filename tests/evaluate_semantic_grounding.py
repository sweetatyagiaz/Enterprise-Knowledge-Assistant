import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from ai_core.rag import KnowledgeAssistant
from ai_core.settings import get_settings
from tests.evaluate_rag_quality import (
    contains_all_keywords,
    detect_abstention,
    reference_answer_overlap,
)


DATASET_PATH = Path("tests/data/rag_ground_truth.json")
REPORT_DIR = Path("reports/evaluations")

VALID_VERDICTS = {
    "SUPPORTED",
    "CONTRADICTED",
    "NOT_ENOUGH_EVIDENCE",
}


JUDGE_PROMPT = """
You are a strict semantic grounding evaluator for a RAG system.

Evaluate the assistant's answer using only the supplied retrieved evidence.

Verdicts:
- SUPPORTED: Every material factual claim is supported by the evidence.
  An appropriate abstention is also SUPPORTED when the evidence cannot answer
  the question and the response makes no unsupported factual claims.
- CONTRADICTED: At least one material factual claim directly conflicts with
  explicit evidence. Examples include a different numeric value, a different
  model name, or a claim explicitly prohibited by the evidence.
- NOT_ENOUGH_EVIDENCE: At least one material claim is neither supported nor
  directly contradicted by the evidence.

Critical distinction between missing and contradictory evidence:
1. An item being absent from a list does NOT automatically mean it is prohibited.
2. If the evidence lists supported options but does not say the list is
   exhaustive, an additional unlisted option is NOT_ENOUGH_EVIDENCE unless
   other evidence directly rules it out.
3. If the evidence explicitly says a format, action, or option is prohibited,
   a claim that it is allowed is CONTRADICTED.
4. Do not treat missing information as proof of a contradiction.
5. If an answer contains both supported and unsupported claims, do not label
   the entire answer SUPPORTED. Use CONTRADICTED only for a direct conflict;
   otherwise use NOT_ENOUGH_EVIDENCE.
6. An abstention is CONTRADICTED if the evidence clearly answers the question
   but the response explicitly claims the answer cannot be found.
7. Do not use general knowledge or assumptions to fill gaps.
8. If the evidence is empty, factual claims normally receive
   NOT_ENOUGH_EVIDENCE.
9. Be conservative and explain which evidence supports or conflicts with
   the answer.

Return only valid JSON with these keys:
{
  "verdict": "SUPPORTED | CONTRADICTED | NOT_ENOUGH_EVIDENCE",
  "rationale": "Brief explanation grounded in the evidence",
  "unsupported_claims": ["claim, if any"]
}
"""


def judge_answer(
    client: OpenAI,
    model: str,
    question: str,
    answer: str,
    sources: list[dict],
) -> dict:
    evidence = []

    for index, source in enumerate(sources, start=1):
        evidence.append(
            {
                "source_number": index,
                "filename": source.get("filename", "Unknown source"),
                "text": source.get("text", ""),
            }
        )

    payload = {
        "question": question,
        "assistant_answer": answer,
        "retrieved_evidence": evidence,
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The judge returned an empty response.")

    result = json.loads(content)
    verdict = str(result.get("verdict", "")).upper().strip()

    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid judge verdict: {verdict!r}")

    unsupported_claims = result.get("unsupported_claims", [])

    if not isinstance(unsupported_claims, list):
        unsupported_claims = []

    return {
        "verdict": verdict,
        "rationale": str(result.get("rationale", "")),
        "unsupported_claims": unsupported_claims,
    }



def evaluate_answer_quality(answer: str, reference: dict) -> dict:
    """Evaluate answer overlap and answerability behavior separately from grounding."""
    answerable = bool(reference.get("answerable"))
    keywords = reference.get("expected_keywords") or []

    all_keywords_present = (
        contains_all_keywords(answer, keywords) if keywords else None
    )
    abstention_detected = detect_abstention(answer)

    if answerable:
        behavior_check_passed = (
            all_keywords_present if keywords else None
        )
        behavior_reason = (
            "All expected keywords appear in the answer."
            if behavior_check_passed is True
            else (
                "One or more expected keywords are absent from the answer."
                if behavior_check_passed is False
                else "No expected keywords are available for this case."
            )
        )
    else:
        behavior_check_passed = abstention_detected
        behavior_reason = (
            "Recognized an abstention phrase for an unanswerable question."
            if abstention_detected
            else "No recognized abstention phrase for an unanswerable question."
        )

    return {
        "reference_answer_comparison": reference_answer_overlap(
            answer,
            reference.get("reference_answer"),
        ),
        "answer_check": {
            "answerable_expected": answerable,
            "expected_keywords": keywords,
            "all_keywords_in_answer": all_keywords_present,
            "abstention_detected": abstention_detected,
            "behavior_check_passed": behavior_check_passed,
            "behavior_check_reason": behavior_reason,
        },
    }


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError(
            "OpenAI API key is missing. Check your .env configuration."
        )

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    client = OpenAI(api_key=settings.openai_api_key)
    assistant = KnowledgeAssistant()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = REPORT_DIR / (
        f"semantic_grounding_{timestamp}.json"
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    errors = []

    print("=" * 65)
    print("SEMANTIC GROUNDING EVALUATION")
    print("=" * 65)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Judge model: {settings.chat_model}")
    print(f"Cases: {len(dataset)}")

    for item in dataset:
        case_id = item.get("id", "UNKNOWN")
        question = item["question"]

        print(f"\n[{case_id}] {question}")

        try:
            rag_result = assistant.ask(question)
            answer = rag_result.get("answer") or ""
            sources = rag_result.get("sources") or []

            judgment = judge_answer(
                client=client,
                model=settings.chat_model,
                question=question,
                answer=answer,
                sources=sources,
            )
            answer_quality = evaluate_answer_quality(answer, item)

            record = {
                "id": case_id,
                "question": question,
                "answerable_expected": item.get("answerable"),
                "answer": answer,
                **answer_quality,
                "source_count": len(sources),
                "sources": [
                    {
                        "filename": source.get(
                            "filename", "Unknown source"
                        ),
                        "score": source.get("score"),
                    }
                    for source in sources
                ],
                **judgment,
            }

            results.append(record)

            print(f"  Verdict: {judgment['verdict']}")
            print(f"  Sources: {len(sources)}")
            print(f"  Rationale: {judgment['rationale']}")

            if judgment["unsupported_claims"]:
                print(
                    "  Unsupported claims: "
                    f"{judgment['unsupported_claims']}"
                )

        except Exception as exc:
            error_record = {
                "id": case_id,
                "question": question,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            errors.append(error_record)

            print(f"  ERROR: {type(exc).__name__}: {exc}")

    counts = Counter(record["verdict"] for record in results)
    completed = len(results)
    supported = counts["SUPPORTED"]

    answerable_results = [
        record for record in results
        if record.get("answerable_expected") is True
    ]
    unanswerable_results = [
        record for record in results
        if record.get("answerable_expected") is False
    ]

    lexical_f1_values = [
        record["reference_answer_comparison"]["f1"]
        for record in answerable_results
        if record.get("reference_answer_comparison", {}).get("f1") is not None
    ]
    behavior_checks = [
        record["answer_check"]["behavior_check_passed"]
        for record in results
        if record.get("answer_check", {}).get("behavior_check_passed") is not None
    ]

    summary_metrics = {
        "answerable_cases": len(answerable_results),
        "unanswerable_cases": len(unanswerable_results),
        "answerable_mean_reference_lexical_f1": (
            round(sum(lexical_f1_values) / len(lexical_f1_values), 3)
            if lexical_f1_values else None
        ),
        "answerable_cases_with_reference_score": len(lexical_f1_values),
        "answer_behavior_pass_rate": (
            round(
                sum(value is True for value in behavior_checks)
                / len(behavior_checks),
                3,
            )
            if behavior_checks else None
        ),
        "answer_behavior_checks_completed": len(behavior_checks),
    }

    report = {
        "evaluation": "semantic_grounding",
        "timestamp_utc": timestamp,
        "judge_model": settings.chat_model,
        "dataset_path": str(DATASET_PATH),
        "total_cases": len(dataset),
        "completed_cases": completed,
        "failed_cases": len(errors),
        "verdict_counts": {
            verdict: counts[verdict]
            for verdict in sorted(VALID_VERDICTS)
        },
        "supported_rate": (
            supported / completed if completed else None
        ),
        "answer_quality_summary": summary_metrics,
        "results": results,
        "errors": errors,
        "limitations": [
            "This is an LLM-as-judge evaluation, not a formal proof "
            "of factual correctness.",
            "Judge decisions may be inconsistent or biased.",
            "Results should be reviewed against manually labeled examples.",
            "The supported rate is the fraction of completed cases judged "
            "SUPPORTED; it is not an overall accuracy score.",
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("SEMANTIC GROUNDING SUMMARY")
    print("=" * 65)
    print(f"Completed: {completed}/{len(dataset)}")
    print(f"Failed: {len(errors)}")

    for verdict in sorted(VALID_VERDICTS):
        print(f"{verdict}: {counts[verdict]}")

    if completed:
        print(f"Supported rate: {supported / completed:.1%}")

    print("\nANSWER QUALITY SUMMARY")
    print("-" * 65)
    print(
        "Answerable cases: "
        f"{summary_metrics['answerable_cases']}"
    )
    print(
        "Unanswerable cases: "
        f"{summary_metrics['unanswerable_cases']}"
    )

    lexical_f1 = summary_metrics[
        "answerable_mean_reference_lexical_f1"
    ]
    if lexical_f1 is not None:
        print(
            "Mean reference lexical F1 (answerable cases): "
            f"{lexical_f1:.3f}"
        )
    else:
        print("Mean reference lexical F1: unavailable")

    behavior_rate = summary_metrics["answer_behavior_pass_rate"]
    if behavior_rate is not None:
        print(f"Answer behavior pass rate: {behavior_rate:.1%}")
    else:
        print("Answer behavior pass rate: unavailable")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
