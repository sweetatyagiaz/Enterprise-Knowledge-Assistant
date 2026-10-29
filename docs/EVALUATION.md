# Evaluation methodology and recorded results

This document summarizes the recorded Sprint 20 evaluation. Results should be regenerated after meaningful changes to retrieval, chunking, embeddings, prompts, model configuration, or indexed content.

## Recorded Sprint 20 snapshot

- Automated tests: **309 passed** in the recorded full-suite run.
- Reviewed relevance cases: **10 answerable cases**.
- Reviewed chunk labels: **19**.
- Indexed chunks in that run: **23**.
- Evaluation cases in the broader report: **18** (14 answerable, 4 unanswerable).

### Human-reviewed chunk relevance

| Metric | Recorded value | Interpretation |
|---|---:|---|
| Precision@5 | 0.32 | Share of the top five retrieved positions judged relevant under the reviewed labels |
| Recall@5 | 0.85 | Share of the reviewed relevant chunks retrieved in the top five |
| MRR | 0.95 | Ranking of the first relevant result, averaged across reviewed cases |
| nDCG@5 | 0.8223 | Ranking quality with graded relevance, discounted by position |

These figures are tied to a particular corpus, set of labels, and evaluation run. They should not be generalized to arbitrary documents or users.

## Important interpretation notes

- The reviewed label set contains only the chunks that were explicitly annotated. Recall is relative to known reviewed labels, not exhaustive corpus-wide recall.
- The local file `tests/data/retrieval_relevance_labels.json` is intentionally excluded from Git. Do not publish it unless it has been reviewed and approved as a public fixture.
- Chunk IDs may change when documents are re-ingested or the SQLite index is rebuilt. The evaluation workflow validates labels against the current index and should fail clearly when labels become stale.
- Human-reviewed relevance metrics and keyword-proxy metrics measure different things. Keep them separate in reports.
- Automated test success does not prove factual correctness, security, fairness, or production readiness.
- The values above are the recorded snapshot from 29 October 2026. Re-run the evaluation to report a newer result.

## Reproducing the checks

From the repository root, with dependencies and local configuration installed:

```bash
PYTHONPATH=. python -m pytest -q
```

Run the retrieval evaluation using the command documented by the repository's current evaluation script. In the recorded workflow this was:

```bash
PYTHONPATH=. python tests/evaluate_retrieval.py
```

Review the generated report under `reports/evaluations/`. Reports can contain document names, queries, retrieved snippets, or other data; inspect them before committing or publishing.

## Metric definitions

- **Precision@K:** Relevant retrieved results divided by the number of results returned in the top K positions, using the evaluator's relevance-label policy.
- **Recall@K:** Relevant labeled chunks retrieved in the top K divided by all relevant chunks labeled for that case.
- **MRR:** Mean reciprocal rank of the first relevant result.
- **nDCG@K:** Normalized discounted cumulative gain, a ranking metric that gives higher value to relevant items appearing earlier and can account for graded relevance.

## Recommended next evaluation improvements

- Increase the number and diversity of reviewed cases.
- Document annotation instructions and adjudicate ambiguous labels.
- Include multiple document types and realistic paraphrases.
- Track regressions across a fixed, versioned public dataset.
- Report latency and memory benchmarks with environment details and repeated runs.
