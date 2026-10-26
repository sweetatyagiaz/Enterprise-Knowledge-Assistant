"""Evaluate grounding behavior against a labeled adversarial dataset."""

import json
from pathlib import Path

from tests.evaluate_grounding import evaluate_claim

DATASET_PATH = Path("tests/grounding_adversarial_cases.json")


def main() -> int:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    total = len(cases)
    correct = 0
    false_acceptances = 0
    false_rejections = 0
    label_mismatches = []

    print("=== Adversarial Grounding Evaluation ===")

    for case in cases:
        result = evaluate_claim(case["claim"], case["sources"])
        expected = case["expected"]
        actual = result["label"]
        passed = actual == expected

        if passed:
            correct += 1

        # An unsupported claim was incorrectly accepted as supported.
        if expected == "UNSUPPORTED" and actual == "SUPPORTED":
            false_acceptances += 1

        # A supported claim was incorrectly rejected or marked uncertain.
        if expected == "SUPPORTED" and actual != "SUPPORTED":
            false_rejections += 1

        if not passed:
            label_mismatches.append(
                {
                    "id": case["id"],
                    "expected": expected,
                    "actual": actual,
                    "reason": result.get("reason"),
                }
            )

        status = "PASS" if passed else "FAIL"
        print(
            f"[{status}] {case['id']}: "
            f"expected={expected}, actual={actual}"
        )

    accuracy = correct / total if total else 0.0

    print("\n=== Summary ===")
    print(f"Cases evaluated: {total}")
    print(f"Correct labels: {correct}")
    print(f"Label mismatches: {len(label_mismatches)}")
    print(f"Label accuracy: {accuracy:.1%}")
    print(f"False acceptances: {false_acceptances}")
    print(f"False rejections: {false_rejections}")

    report = {
        "dataset": str(DATASET_PATH),
        "cases_evaluated": total,
        "correct_labels": correct,
        "label_mismatches": len(label_mismatches),
        "label_accuracy": round(accuracy, 4),
        "false_acceptances": false_acceptances,
        "false_rejections": false_rejections,
        "mismatches": label_mismatches,
        "note": (
            "Results measure agreement with this hand-labeled dataset, "
            "not general semantic grounding accuracy."
        ),
    }

    output_path = Path("reports/evaluations/adversarial_grounding_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Report saved: {output_path}")

    return 0 if not label_mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
