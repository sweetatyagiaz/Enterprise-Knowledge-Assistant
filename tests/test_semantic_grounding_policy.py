from tests.evaluate_semantic_grounding import JUDGE_PROMPT


def test_unlisted_options_are_not_automatically_contradictions():
    prompt = JUDGE_PROMPT.lower()

    assert "does not automatically mean it is prohibited" in prompt
    assert "not_enough_evidence" in prompt
    assert "directly contradicted" in prompt


def test_explicit_prohibition_is_a_contradiction():
    prompt = JUDGE_PROMPT.lower()

    assert "explicitly says a format, action, or option is prohibited" in prompt
    assert "contradicted" in prompt
