
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from ai_core.settings import get_settings
from ai_core.retrieval.grounding_evaluator import evaluate_grounding
from tests.evaluate_semantic_grounding import judge_answer


DATASET_PATH = Path("tests/data/semantic_grounding_adversarial.json")
REPORT_DIR = Path("reports/evaluations")
REPEATS = 3


def main():
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key is missing.")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    client = OpenAI(api_key=settings.openai_api_key)

    results = []
    errors = []
    verdicts_by_case = defaultdict(list)
    sources_by_case = defaultdict(list)

    print("=" * 65)
    print("SEMANTIC GROUNDING STABILITY EVALUATION")
    print("=" * 65)
    print(f"Cases per run: {len(dataset)}")
    print(f"Repeated runs: {REPEATS}")
    print(f"Maximum evaluations: {len(dataset) * REPEATS}")
    print(f"Judge model: {settings.chat_model}")

    for run_number in range(1, REPEATS + 1):
        print(f"\n--- Run {run_number}/{REPEATS} ---")

        for item in dataset:
            case_id = item["id"]

            try:
                def llm_judge():
                    return judge_answer(
                        client=client,
                        model=settings.chat_model,
                        question=item["question"],
                        answer=item["answer"],
                        sources=[
                            {
                                "filename": "adversarial_test_evidence.txt",
                                "text": item["evidence"],
                            }
                        ],
                    )

                judgment = evaluate_grounding(
                    claim=item["answer"],
                    evidence=item["evidence"],
                    llm_judge=llm_judge,
                )

                verdict_source = judgment.pop("verdict_source")
                actual = judgment["verdict"]
                expected = item["expected_verdict"]

                verdicts_by_case[case_id].append(actual)
                sources_by_case[case_id].append(verdict_source)

                record = {
                    "run": run_number,
                    "id": case_id,
                    "expected_verdict": expected,
                    "actual_verdict": actual,
                    "correct": actual == expected,
                    "verdict_source": verdict_source,
                    "rationale": judgment["rationale"],
                }
                results.append(record)

                print(
                    f"{case_id}: {actual} "
                    f"({'PASS' if actual == expected else 'FAIL'}) "
                    f"[{verdict_source}]"
                )

            except Exception as exc:
                errors.append(
                    {
                        "run": run_number,
                        "id": case_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"{case_id}: ERROR - "
                    f"{type(exc).__name__}: {exc}"
                )

    case_summaries = []

    for item in dataset:
        case_id = item["id"]
        observed = verdicts_by_case.get(case_id, [])
        counts = Counter(observed)
        sources = Counter(sources_by_case.get(case_id, []))

        case_summaries.append(
            {
                "id": case_id,
                "expected_verdict": item["expected_verdict"],
                "observed_runs": len(observed),
                "verdict_counts": dict(counts),
                "verdict_sources": dict(sources),
                "consistent": (
                    len(counts) == 1 and len(observed) == REPEATS
                ),
                "modal_verdict": (
                    counts.most_common(1)[0][0] if counts else None
                ),
            }
        )

    completed = len(results)
    correct = sum(record["correct"] for record in results)
    consistent_cases = sum(
        case["consistent"] for case in case_summaries
    )
    total_expected_evaluations = len(dataset) * REPEATS
    source_counts = Counter(
        record["verdict_source"] for record in results
    )

    report = {
        "evaluation": "semantic_grounding_stability",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "judge_model": settings.chat_model,
        "repeats": REPEATS,
        "cases_per_run": len(dataset),
        "expected_evaluations": total_expected_evaluations,
        "completed_evaluations": completed,
        "verdict_source_counts": dict(source_counts),
        "errors": errors,
        "overall_label_accuracy": (
            correct / completed if completed else None
        ),
        "case_consistency_rate": (
            consistent_cases / len(dataset) if dataset else None
        ),
        "consistent_cases": consistent_cases,
        "case_count": len(dataset),
        "case_summaries": case_summaries,
        "results": results,
        "limitations": [
            "Repeated identical inputs measure observed verdict stability, "
            "not factual correctness.",
            "A consistent verdict can still be wrong.",
            "Model behavior may change across model versions or service updates.",
            "Deterministic policy cases do not measure LLM judge stability.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = (
        REPORT_DIR / "semantic_grounding_stability_report.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("STABILITY SUMMARY")
    print("=" * 65)
    print(
        f"Completed evaluations: "
        f"{completed}/{total_expected_evaluations}"
    )
    print(f"Processing errors: {len(errors)}")
    print(f"Verdict sources: {dict(source_counts)}")

    if completed:
        print(f"Label accuracy: {correct / completed:.1%}")

    if dataset:
        print(
            f"Case consistency: {consistent_cases}/{len(dataset)} "
            f"({consistent_cases / len(dataset):.1%})"
        )

    print("\nCases with varying verdicts:")
    variable_cases = [
        case for case in case_summaries if not case["consistent"]
    ]

    if variable_cases:
        for case in variable_cases:
            print(f"{case['id']}: {case['verdict_counts']}")
    else:
        print("None")

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
