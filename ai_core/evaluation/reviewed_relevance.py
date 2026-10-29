"""Evaluate retrieval rankings against reviewed chunk-level relevance labels."""

import json
import math
from pathlib import Path
from typing import Any

from ai_core.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def load_relevance_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load reviewed labels, rejecting duplicate cases and invalid grades."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError("Relevance-label dataset must be a JSON list.")

    labels = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each label record requires a non-empty case_id.")

        if case_id in labels:
            raise ValueError(f"Duplicate relevance-label case: {case_id}")

        if record.get("reviewed") is not True:
            raise ValueError(f"Labels for {case_id} are not reviewed.")

        chunks = record.get("relevant_chunks")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"{case_id} must contain relevant_chunks.")

        seen_ids = set()
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            grade = chunk.get("grade")
            evidence = chunk.get("evidence")

            if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
                raise ValueError(f"{case_id} contains an invalid chunk_id.")
            if chunk_id in seen_ids:
                raise ValueError(f"{case_id} has duplicate chunk ID {chunk_id}.")
            seen_ids.add(chunk_id)

            if (
                isinstance(grade, bool)
                or not isinstance(grade, (int, float))
                or not math.isfinite(grade)
                or grade <= 0
            ):
                raise ValueError(f"{case_id} has an invalid relevance grade.")
            if not isinstance(evidence, str) or not evidence.strip():
                raise ValueError(f"{case_id} has empty evidence.")

        labels[case_id] = record

    return labels


def validate_chunk_labels(
    labels: dict[str, dict[str, Any]],
    indexed_chunks: list[dict[str, Any]],
) -> None:
    """Ensure labeled IDs still point to chunks containing their evidence.

    Evidence is validated using explicit evidence_terms when supplied.
    Otherwise, the stored chunk must match its recorded filename and ID;
    natural-language evidence descriptions are not treated as exact quotes.
    """
    indexed_by_id = {}
    for chunk in indexed_chunks:
        chunk_id = chunk.get("id")
        if chunk_id in indexed_by_id:
            raise ValueError(f"Duplicate indexed chunk ID: {chunk_id}")
        indexed_by_id[chunk_id] = chunk

    for case_id, record in labels.items():
        for label in record["relevant_chunks"]:
            chunk_id = label["chunk_id"]
            chunk = indexed_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(
                    f"{case_id}: labeled chunk ID {chunk_id} is missing."
                )

            expected_filename = label.get("filename", "README.md")
            if chunk.get("filename") != expected_filename:
                raise ValueError(
                    f"{case_id}: chunk {chunk_id} filename mismatch."
                )

            evidence_terms = label.get("evidence_terms", [])
            text = chunk.get("text", chunk.get("chunk_text", ""))
            missing_terms = [
                term for term in evidence_terms
                if term.casefold() not in text.casefold()
            ]
            if missing_terms:
                raise ValueError(
                    f"{case_id}: chunk {chunk_id} lacks evidence terms: "
                    f"{missing_terms}"
                )


def evaluate_reviewed_relevance(
    ranked_ids: list[str | int],
    label_record: dict[str, Any] | None,
    k: int = 5,
) -> dict[str, Any] | None:
    """Calculate metrics for a case with reviewed chunk relevance labels."""
    if label_record is None or label_record.get("reviewed") is not True:
        return None

    relevance = {
        str(item["chunk_id"]): float(item["grade"])
        for item in label_record["relevant_chunks"]
    }
    ranked = [str(chunk_id) for chunk_id in ranked_ids]
    relevant_ids = set(relevance)

    return {
        "precision_at_k": precision_at_k(ranked, relevant_ids, k),
        "recall_at_k": recall_at_k(ranked, relevant_ids, k),
        "mrr": reciprocal_rank(ranked, relevant_ids),
        "ndcg_at_k": ndcg_at_k(ranked, relevance, k),
        "known_relevant_chunk_count": len(relevant_ids),
        "relevance_judgment": "reviewed_chunk_labels",
    }
