
import re
from collections import Counter


def _tokens(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def hybrid_rank(
    query: str,
    results: list[dict],
    semantic_weight: float = 0.8,
    lexical_weight: float = 0.2,
) -> list[dict]:
    """Rerank semantic results using lexical overlap as a secondary signal."""

    if semantic_weight < 0 or lexical_weight < 0:
        raise ValueError("Ranking weights must be non-negative.")

    if semantic_weight + lexical_weight == 0:
        raise ValueError("At least one ranking weight must be positive.")

    query_tokens = set(_tokens(query))
    if not results or not query_tokens:
        return results

    reranked = []

    for result in results:
        chunk_tokens = _tokens(result.get("text", ""))
        chunk_counts = Counter(chunk_tokens)

        matched_tokens = query_tokens.intersection(chunk_counts)
        lexical_score = len(matched_tokens) / len(query_tokens)

        item = dict(result)
        item["semantic_score"] = float(result.get("score", 0.0))
        item["lexical_score"] = lexical_score
        item["hybrid_score"] = (
            semantic_weight * item["semantic_score"]
            + lexical_weight * lexical_score
        )

        reranked.append(item)

    reranked.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )
    return reranked

