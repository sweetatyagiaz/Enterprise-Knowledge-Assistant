
import json
from pathlib import Path

from ai_core.rag import KnowledgeAssistant


DATASET_PATH = Path("tests/data/rag_eval.json")


def check_expected_keywords(answer: str, keywords: list[str]) -> bool | None:
    if not keywords:
        return None

    answer_lower = answer.lower()
    return all(keyword.lower() in answer_lower for keyword in keywords)


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assistant = KnowledgeAssistant()

    answerable_count = 0
    answer_keyword_hits = 0
    abstention_checks = 0
    abstention_hits = 0

    for item in dataset:
        question = item["question"]

        print(f"\n[{item['id']}] {question}")

        try:
            result = assistant.ask(question)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue

        answer = result.get("answer") or ""
        sources = result.get("sources") or []
        keywords = item.get("expected_keywords", [])

        keyword_hit = check_expected_keywords(answer, keywords)

        print(f"  Answer: {answer}")
        print(f"  Sources returned: {len(sources)}")

        for source in sources[:3]:
            print(
                f"    - {source.get('filename', 'Unknown source')} "
                f"(score={source.get('score', 0):.3f})"
            )

        if item.get("answerable", False):
            answerable_count += 1
            if keyword_hit is not None:
                answer_keyword_hits += int(keyword_hit)

            print(f"  Expected keywords in answer: {keyword_hit}")
        else:
            abstention_checks += 1
            answer_lower = answer.lower()

            abstained = any(
                phrase in answer_lower
                for phrase in (
                    "does not contain",
                    "do not contain",
                    "not provided",
                    "cannot determine",
                    "can't determine",
                    "don't know",
                    "not enough information",
                    "could not find",
                    "unable to determine",
                )
            )

            abstention_hits += int(abstained)
            print(f"  Abstention detected: {abstained}")

    print("\n=== Live Answer Evaluation ===")

    if answerable_count:
        print(
            "Answer keyword pass rate: "
            f"{answer_keyword_hits / answerable_count:.1%}"
        )

    if abstention_checks:
        print(
            "Abstention pass rate: "
            f"{abstention_hits / abstention_checks:.1%}"
        )

    print(
        "\nNote: keyword checks and phrase-based abstention detection "
        "are simple heuristics, not a full factuality evaluation."
    )


if __name__ == "__main__":
    main()

