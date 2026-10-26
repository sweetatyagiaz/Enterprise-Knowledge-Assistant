
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests


DATASET_PATH = Path("tests/data/rag_eval.json")
REPORT_DIR = Path("reports/evaluations")
API_URL = "http://127.0.0.1:8000/ask"
TIMEOUT_SECONDS = 90

ABSTENTION_PHRASES = (
    "does not contain",
    "does not include",
    "do not contain",
    "not provided",
    "cannot determine",
    "can't determine",
    "don't know",
    "not enough information",
    "could not find",
    "unable to determine",
    "no information about",
)


def check_expected_keywords(
    answer: str,
    keywords: list[str],
) -> bool | None:
    if not keywords:
        return None

    answer_lower = answer.casefold()
    return all(
        keyword.casefold() in answer_lower
        for keyword in keywords
    )


def detects_abstention(answer: str) -> bool:
    answer_lower = answer.casefold()
    return any(
        phrase in answer_lower
        for phrase in ABSTENTION_PHRASES
    )


def main() -> int:
    dataset = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    timestamp = datetime.now(timezone.utc)
    report = {
        "timestamp_utc": timestamp.isoformat(),
        "api_url": API_URL,
        "dataset": str(DATASET_PATH),
        "results": [],
        "summary": {},
        "limitations": (
            "Keyword matching does not prove factual correctness. "
            "Phrase-based abstention detection is a heuristic."
        ),
    }

    passed_answerable = 0
    failed_answerable = 0
    passed_unanswerable = 0
    failed_unanswerable = 0
    skipped_answerable = 0

    for item in dataset:
        question = item["question"]
        answerable = item.get("answerable", False)

        print(f"\n[{item['id']}] {question}")

        result_record = {
            "id": item["id"],
            "question": question,
            "expected_answerable": answerable,
            "expected_keywords": item.get(
                "expected_keywords", []
            ),
            "answer": None,
            "sources_count": 0,
            "keyword_check": None,
            "abstention_detected": None,
            "status": "FAIL",
            "error": None,
        }

        try:
            response = requests.post(
                API_URL,
                json={"question": question, "top_k": 5},
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            result_record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(f"  ERROR: {result_record['error']}")

            if answerable:
                failed_answerable += 1
            else:
                failed_unanswerable += 1

            report["results"].append(result_record)
            continue

        answer = payload.get("answer")
        sources = payload.get("sources") or []

        result_record["answer"] = answer
        result_record["sources_count"] = len(sources)
        result_record["sources"] = [
            {
                "filename": source.get("filename", "Unknown source"),
                "text": source.get("text", ""),
                "score": source.get("score"),
                "hybrid_score": source.get("hybrid_score"),
            }
            for source in sources
        ]

        if not isinstance(answer, str) or not answer.strip():
            result_record["error"] = (
                "Response has no non-empty answer."
            )
            print(f"  ERROR: {result_record['error']}")

            if answerable:
                failed_answerable += 1
            else:
                failed_unanswerable += 1

            report["results"].append(result_record)
            continue

        print(f"  Answer: {answer}")
        print(f"  Sources returned: {len(sources)}")

        if answerable:
            keyword_result = check_expected_keywords(
                answer,
                item.get("expected_keywords", []),
            )
            result_record["keyword_check"] = keyword_result

            if keyword_result is None:
                result_record["status"] = "SKIP"
                skipped_answerable += 1
                print("  SKIP: No expected keywords configured.")
            elif keyword_result:
                result_record["status"] = "PASS"
                passed_answerable += 1
                print("  Expected keywords present: True")
                print("  RESULT: PASS")
            else:
                failed_answerable += 1
                print("  Expected keywords present: False")
                print("  RESULT: FAIL")
        else:
            abstained = detects_abstention(answer)
            result_record["abstention_detected"] = abstained

            if abstained:
                result_record["status"] = "PASS"
                passed_unanswerable += 1
                print("  Abstention detected: True")
                print("  RESULT: PASS")
            else:
                failed_unanswerable += 1
                print("  Abstention detected: False")
                print("  RESULT: FAIL")

        report["results"].append(result_record)

    answerable_total = passed_answerable + failed_answerable
    unanswerable_total = (
        passed_unanswerable + failed_unanswerable
    )
    scored_total = answerable_total + unanswerable_total
    total_passed = passed_answerable + passed_unanswerable
    total_failed = failed_answerable + failed_unanswerable

    report["summary"] = {
        "total_dataset_cases": len(dataset),
        "answerable_passed": passed_answerable,
        "answerable_failed": failed_answerable,
        "answerable_skipped": skipped_answerable,
        "unanswerable_passed": passed_unanswerable,
        "unanswerable_failed": failed_unanswerable,
        "total_scored": scored_total,
        "total_passed": total_passed,
        "total_failed": total_failed,
        "answerable_keyword_pass_rate": (
            passed_answerable / answerable_total
            if answerable_total else None
        ),
        "abstention_pass_rate": (
            passed_unanswerable / unanswerable_total
            if unanswerable_total else None
        ),
        "overall_pass_rate": (
            total_passed / scored_total
            if scored_total else None
        ),
    }

    print("\n=== Live API Evaluation Summary ===")
    print(
        f"Answerable: {passed_answerable}/{answerable_total} passed"
    )
    print(
        f"Unanswerable: "
        f"{passed_unanswerable}/{unanswerable_total} passed"
    )
    print(f"Skipped answerable cases: {skipped_answerable}")
    print(f"Overall: {total_passed}/{scored_total} passed")
    print(f"Failed: {total_failed}")

    if answerable_total:
        print(
            "Answerable keyword pass rate: "
            f"{passed_answerable / answerable_total:.1%}"
        )

    if unanswerable_total:
        print(
            "Abstention pass rate: "
            f"{passed_unanswerable / unanswerable_total:.1%}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / (
        f"live_api_evaluation_"
        f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nJSON report saved: {report_path}")
    print(f"Limitations: {report['limitations']}")

    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
