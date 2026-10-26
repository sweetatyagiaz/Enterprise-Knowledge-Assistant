
import json
from collections import Counter
from pathlib import Path

from openai import OpenAI

from ai_core.settings import get_settings
from ai_core.retrieval.grounding_evaluator import evaluate_grounding
from tests.evaluate_semantic_grounding import judge_answer


DATASET_PATH = Path("tests/data/semantic_grounding_adversarial.json")

VERDICTS = [
    "SUPPORTED",
    "CONTRADICTED",
    "NOT_ENOUGH_EVIDENCE",
]


def calculate_classification_metrics(results):
    """Build a confusion matrix and per-verdict classification metrics."""
    matrix = {
        expected: {actual: 0 for actual in VERDICTS}
        for expected in VERDICTS
    }

    for item in results:
        expected = item["expected_verdict"]
        actual = item["actual_verdict"]

        if expected in matrix and actual in matrix[expected]:
            matrix[expected][actual] += 1

    metrics = {}

    for verdict in VERDICTS:
        true_positive = matrix[verdict][verdict]

        false_positive = sum(
            matrix[other][verdict]
            for other in VERDICTS
            if other != verdict
        )

        false_negative = sum(
            matrix[verdict][other]
            for other in VERDICTS
            if other != verdict
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else None
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator
            else None
        )

        if (
            precision is not None
            and recall is not None
            and precision + recall > 0
        ):
            f1 = 2 * precision * recall / (precision + recall)
        elif precision == 0 or recall == 0:
            f1 = 0.0
        else:
            f1 = None

        metrics[verdict] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(matrix[verdict].values()),
        }

    return matrix, metrics


def main():
    settings = get_settings()

    if not settings.openai_api_key:
        raise RuntimeError("OpenAI API key is missing.")

    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    client = OpenAI(api_key=settings.openai_api_key)

    results = []
    errors = []

    print("=" * 65)
    print("ADVERSARIAL SEMANTIC GROUNDING EVALUATION")
    print("=" * 65)
    print(f"Cases: {len(dataset)}")
    print(f"Judge model: {settings.chat_model}")

    for item in dataset:
        case_id = item["id"]

        try:
            # Called only when the deterministic policy cannot decide.
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

            expected = item["expected_verdict"]
            actual = judgment["verdict"]
            passed = actual == expected

            record = {
                "id": case_id,
                "question": item["question"],
                "answer": item["answer"],
                "evidence": item["evidence"],
                "expected_verdict": expected,
                "actual_verdict": actual,
                "passed": passed,
                "verdict_source": verdict_source,
                "rationale": judgment["rationale"],
                "unsupported_claims": judgment["unsupported_claims"],
            }

            results.append(record)

            status = "PASS" if passed else "FAIL"

            print(f"\n[{case_id}] {status}")
            print(f"  Expected: {expected}")
            print(f"  Actual:   {actual}")
            print(f"  Verdict source: {verdict_source}")
            print(f"  Rationale: {judgment['rationale']}")

        except Exception as exc:
            error = {
                "id": case_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            errors.append(error)

            print(
                f"\n[{case_id}] ERROR: "
                f"{type(exc).__name__}: {exc}"
            )

    total = len(dataset)
    completed = len(results)

    passed_count = sum(item["passed"] for item in results)

    counts = Counter(
        item["actual_verdict"] for item in results
    )

    source_counts = Counter(
        item["verdict_source"] for item in results
    )

    confusion_matrix, class_metrics = (
        calculate_classification_metrics(results)
    )

    report = {
        "evaluation": "adversarial_semantic_grounding",
        "judge_model": settings.chat_model,
        "total_cases": total,
        "completed_cases": completed,
        "failed_api_or_processing_cases": len(errors),
        "correct_verdicts": passed_count,
        "incorrect_verdicts": completed - passed_count,
        "accuracy": (
            passed_count / completed if completed else None
        ),
        "verdict_counts": dict(counts),
        "verdict_source_counts": dict(source_counts),
        "confusion_matrix": confusion_matrix,
        "classification_metrics": class_metrics,
        "results": results,
        "errors": errors,
        "limitations": [
            "Results measure agreement with the hand-labeled test cases.",
            "LLM judgments can be inconsistent and should be manually reviewed.",
            "A small adversarial dataset does not establish production reliability.",
            "Deterministic policy coverage is narrow and should be tested separately.",
        ],
    }

    report_dir = Path("reports/evaluations")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = (
        report_dir / "semantic_grounding_adversarial_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("ADVERSARIAL SUMMARY")
    print("=" * 65)

    print(f"Completed: {completed}/{total}")
    print(f"Correct verdicts: {passed_count}/{completed}")
    print(f"Incorrect verdicts: {completed - passed_count}")
    print(f"Processing errors: {len(errors)}")
    print(f"Verdict sources: {dict(source_counts)}")

    print("\nConfusion matrix (rows=expected, columns=actual):")

    print(
        f"{'EXPECTED / ACTUAL':<25}"
        + "".join(
            f"{verdict:>25}"
            for verdict in VERDICTS
        )
    )

    for expected in VERDICTS:
        print(
            f"{expected:<25}"
            + "".join(
                f"{confusion_matrix[expected][actual]:>25}"
                for actual in VERDICTS
            )
        )

    print("\nPer-verdict metrics:")

    def format_metric(value):
        return f"{value:.1%}" if value is not None else "N/A"

    for verdict in VERDICTS:
        metric = class_metrics[verdict]

        print(
            f"{verdict}: "
            f"precision={format_metric(metric['precision'])}, "
            f"recall={format_metric(metric['recall'])}, "
            f"F1={format_metric(metric['f1'])}, "
            f"support={metric['support']}"
        )

    if completed:
        print(f"Accuracy: {passed_count / completed:.1%}")

    for verdict in sorted({
        item["expected_verdict"] for item in dataset
    }):
        subset = [
            item for item in results
            if item["expected_verdict"] == verdict
        ]

        correct = sum(item["passed"] for item in subset)

        print(f"{verdict}: {correct}/{len(subset)} correct")

    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
