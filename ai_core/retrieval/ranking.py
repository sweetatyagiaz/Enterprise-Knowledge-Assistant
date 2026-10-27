
import re
import math


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "the",
    "this", "to", "use", "uses", "using", "what", "which",
    "with", "does", "do", "default", "project",
}


def _tokens(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _content_tokens(text: str) -> set[str]:
    """Remove common words that carry little retrieval meaning."""
    return {
        token
        for token in _tokens(text)
        if token not in STOP_WORDS
    }


def _phrase_score(query: str, text: str) -> float:
    """Measure whether meaningful adjacent query-token pairs occur."""
    tokens = _tokens(query)
    phrases = [
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
        if left not in STOP_WORDS and right not in STOP_WORDS
    ]

    if not phrases:
        return 0.0

    normalized_text = " ".join(_tokens(text))
    matches = sum(
        phrase in normalized_text
        for phrase in phrases
    )

    return matches / len(phrases)


def hybrid_rank(
    query: str,
    results: list[dict],
    semantic_weight: float = 0.75,
    lexical_weight: float = 0.15,
    phrase_weight: float = 0.10,
) -> list[dict]:
    """
    Rerank semantic candidates using semantic, lexical, and phrase scores.

    Args:
        query: User's search query.
        results: Retrieved chunks containing text and optional semantic scores.
        semantic_weight: Weight assigned to the semantic similarity score.
        lexical_weight: Weight assigned to keyword overlap.
        phrase_weight: Weight assigned to adjacent query-phrase matches.

    Returns:
        A reranked list of copied result dictionaries with:
        semantic_score, lexical_score, phrase_score, and hybrid_score.

    Raises:
        ValueError: If weights are negative or non-finite, all weights are
            zero, or a semantic score is non-finite.
    """
    import math

    weights = (semantic_weight, lexical_weight, phrase_weight)

    if any(not math.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError(
            "Ranking weights must be finite and non-negative."
        )

    if sum(weights) <= 0:
        raise ValueError("At least one ranking weight must be positive.")

    query_tokens = _content_tokens(query)

    if not results or not query_tokens:
        return results

    reranked = []

    for result in results:
        text = result.get("text", "")
        chunk_tokens = _content_tokens(text)

        lexical_score = (
            len(query_tokens.intersection(chunk_tokens))
            / len(query_tokens)
        )

        phrase_score = _phrase_score(query, text)

        semantic_score = float(result.get("score", 0.0))

        if not math.isfinite(semantic_score):
            raise ValueError("Semantic scores must be finite.")

        item = dict(result)
        item["semantic_score"] = semantic_score
        item["lexical_score"] = lexical_score
        item["phrase_score"] = phrase_score
        item["hybrid_score"] = (
            semantic_weight * semantic_score
            + lexical_weight * lexical_score
            + phrase_weight * phrase_score
        )

        reranked.append(item)

    reranked.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return reranked


