from tests.evaluate_grounding import evaluate_result, split_claims


def test_split_claims_handles_prose_and_bullets():
    answer = (
        "The API uses FastAPI.\n"
        "- The vector store uses SQLite.\n"
        "The system supports retrieval; it generates answers."
    )

    claims = split_claims(answer)

    assert "The API uses FastAPI." in claims
    assert "The vector store uses SQLite." in claims
    assert "The system supports retrieval" in claims
    assert "it generates answers." in claims


def test_evaluate_result_marks_supported_claim():
    result = {
        "question": "Which framework does the API use?",
        "answer": "The API uses FastAPI.",
        "sources": [
            {
                "filename": "README.md",
                "text": "The API uses FastAPI for serving requests.",
            }
        ],
    }

    evaluation = evaluate_result(result)

    assert evaluation["summary"]["claim_count"] == 1
    assert evaluation["claims"][0]["label"] == "SUPPORTED"
    assert evaluation["claims"][0]["evidence"]["filename"] == "README.md"


def test_evaluate_result_flags_unsupported_number():
    result = {
        "question": "What is the maximum upload size?",
        "answer": "The maximum upload size is 50 MB.",
        "sources": [
            {
                "filename": "README.md",
                "text": "The maximum upload size is 15 MB.",
            }
        ],
    }

    evaluation = evaluate_result(result)

    assert evaluation["summary"]["unsupported"] == 1
    assert evaluation["claims"][0]["label"] == "UNSUPPORTED"
    assert "50" in evaluation["claims"][0]["missing_numbers"]


def test_evaluate_result_recognizes_abstention():
    result = {
        "question": "What is the company's annual revenue?",
        "answer": (
            "The provided document context does not contain "
            "information about annual revenue."
        ),
        "sources": [],
    }

    evaluation = evaluate_result(result)

    assert evaluation["claims"][0]["label"] == "ABSTENTION"
    assert evaluation["summary"]["claim_count"] == 0


def test_evaluate_result_handles_missing_sources():
    result = {
        "question": "What does the system do?",
        "answer": "The system provides judicial analytics.",
        "sources": [],
    }

    evaluation = evaluate_result(result)

    assert evaluation["summary"]["claim_count"] == 1
    assert evaluation["summary"]["uncertain"] == 1
