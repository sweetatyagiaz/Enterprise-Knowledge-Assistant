# Sprint 1 Completion Report

## Enterprise Knowledge Assistant --- RAG Platform

**Project:** `ai-architect-portfolio`\
**Sprint:** Sprint 1 --- Retrieval-Augmented Generation (RAG)
Foundation\
**Status:** Core validation completed\
**Environment:** Python 3.13.5, pytest 9.1.1, FastAPI, SQLite

------------------------------------------------------------------------

## 1. Executive Summary

Sprint 1 established and validated the foundational components of the
Enterprise Knowledge Assistant. The system supports document ingestion,
text chunking, embedding generation, vector-based retrieval, hybrid
ranking, and question answering grounded in retrieved document context.

The implementation was validated through automated API and unit tests,
retrieval evaluation, and live API checks. After removing a temporary
upload-test record from the SQLite vector store, the complete test suite
passed again.

### Final validation summary

  Metric                                  Result
  ------------------------------------- --------
  Automated tests passed                   34/34
  Total retrieval evaluation cases            12
  Answerable evaluation cases                 10
  Unanswerable evaluation cases                2
  Source hit rate @5                        100%
  Passage evidence rate @1                  100%
  Passage evidence rate @5                  100%
  Evidence mean reciprocal rank (MRR)      1.000
  Keyword coverage @1                       100%
  Keyword coverage @5                       100%

**Conclusion:** The core Sprint 1 implementation has passed its current
validation suite. Production readiness, security, scalability, and
generated-answer factuality require additional evaluation.

## 2. Objectives and Scope

The primary objective was to build a working RAG foundation that can
ingest documents, retrieve relevant evidence, and answer user questions
using the indexed knowledge base.

The sprint covered:

-   Document ingestion and supported-format validation.
-   Text extraction and overlapping chunk generation.
-   Embedding generation and persistent storage.
-   Similarity search and hybrid retrieval ranking.
-   Context-grounded answer generation.
-   API health checks, document listing, uploads, and question
    answering.
-   Automated regression tests and retrieval evaluation.
-   Handling questions for which the indexed documents do not provide an
    answer.

## 3. Architecture and Components

### 3.1 API layer

The FastAPI application in `api/main.py` exposes the following
endpoints:

  Endpoint            Purpose
  ------------------- -------------------------------------------
  `GET /health`       Checks API health
  `GET /documents`    Lists indexed documents
  `POST /documents`   Accepts and indexes document uploads
  `POST /ask`         Answers questions using retrieved context

The assistant is cached through `get_assistant()` to reuse the
initialized `KnowledgeAssistant`.

### 3.2 RAG processing

The principal processing flow is:

**Document → Text extraction → Chunking → Embeddings → SQLite vector
store → Retrieval and hybrid ranking → Context-grounded answer**

The question-answering workflow embeds the user's question, retrieves
candidate chunks, applies hybrid ranking, and passes the selected
context to the chat-completion model.

The answer-generation instructions direct the model to answer from the
supplied context and avoid inventing unsupported facts.

### 3.3 Technology configuration

  Component                      Current configuration
  ------------------------------ -----------------------------
  API framework                  FastAPI
  Embedding model                `text-embedding-3-small`
  Embedding dimensions           1,536
  Chat-completion model          `gpt-4.1-mini`
  Vector persistence             SQLite
  Database file                  `data/vector_store.sqlite3`
  Default retrieved chunks       5
  Text chunk size                900
  Chunk overlap                  150
  Maximum document upload size   15 MB
  Text extraction formats        PDF, DOCX, TXT, Markdown

These values reflect the project's current configuration and are not
performance benchmarks.

## 4. Automated Testing Results

The full test suite was executed after the temporary test-record
cleanup:

``` bash
python -m pytest -v
```

**Result: 34 passed in 0.57 seconds.**

### 4.1 API integration tests --- 12 passed

Validated health checks, document listing, uploads, unsupported and
oversized file rejection, missing-file handling, question answering,
invalid request handling, and service-unavailable behavior.

### 4.2 Answer-quality tests --- 4 passed

Validated grounded architecture answers, the default embedding model,
source information, and abstention for a question about unknown company
revenue.

### 4.3 RAG evaluation and component tests --- 14 passed

Covered chunking, invalid overlap, empty content, Markdown loading,
unsupported and empty documents, vector similarity search,
retrieved-context use, missing retrieval results, irrelevant-context
handling, and Markdown chunking behavior.

### 4.4 Retrieval-ranking tests --- 4 passed

Validated relevant-chunk prioritization, preservation of original
results, invalid weighting rejection, and zero-total-weight rejection.

## 5. Retrieval Evaluation

The evaluation dataset is located at `tests/data/rag_eval.json`. It
contains 12 cases, identified as RAG-001 through RAG-012.

### 5.1 Results

  Evaluation metric            Result
  -------------------------- --------
  Source hit rate @5             100%
  Passage evidence rate @1       100%
  Passage evidence rate @5       100%
  Evidence MRR                  1.000
  Keyword coverage @1            100%
  Keyword coverage @5            100%

All 10 answerable cases retrieved their expected source, with the
expected keyword evidence present in the top-ranked passage. The two
unanswerable cases concerned annual revenue and annual profit.

### 5.2 Interpretation

The results indicate that the current retrieval pipeline performs well
on this project's small, curated evaluation dataset.

However, the evaluation script uses keyword-based evidence checks. These
are proxies for retrieval quality and do not establish that a generated
answer is fully accurate, complete, or free from unsupported claims.

The results should not be interpreted as proof of 100% real-world
retrieval accuracy.

## 6. Live API Validation

The following checks were previously completed successfully.

### Health endpoint

`GET /health` returned:

``` json
{
  "status": "ok",
  "service": "enterprise-knowledge-assistant"
}
```

### Supported question

A question about the default embedding model returned the expected
answer: `text-embedding-3-small`, accompanied by relevant README chunks.

### Unsupported question

A question about the company's annual revenue returned an abstention
stating that the provided document context did not contain the requested
information.

This demonstrates a successful abstention in that particular live test.
The annual-profit case still needs its own generated-answer test.

## 7. Database Cleanup

A temporary upload test had added a document named `rag_upload_test.txt`
to the SQLite vector store.

The record was inspected before deletion. It was identified as row ID
`192`, containing test data about Aurora Analytics.

The cleanup deleted only the matching row:

-   Rows deleted: 1
-   Remaining records with that filename: 0
-   Application code changed: No
-   Other indexed documents intentionally modified: None

The complete automated test suite and retrieval evaluation were
subsequently rerun successfully.

## 8. Known Limitations and Risks

### 8.1 Generated-answer evaluation

The retrieval evaluation measures evidence retrieval rather than
generated-answer factuality. A larger evaluation should check
correctness, completeness, citation support, hallucinations, and
abstention behavior.

### 8.2 Unanswerable questions

The revenue abstention test passed, but a separate test should verify
that the assistant also abstains when asked about unknown annual profit.

### 8.3 Irrelevant retrieval results

Even when an answer is correctly withheld, the system may return
unrelated chunks. A future iteration should assess relevance thresholds,
score calibration, and whether low-confidence results should be excluded
from the generation context.

### 8.4 Evaluation dataset size

Twelve curated cases are useful for regression testing but insufficient
to establish robustness across diverse document types, ambiguous
questions, long documents, and large knowledge bases.

### 8.5 Operational readiness

The current test results do not establish production readiness.
Authentication, authorization, upload security, observability,
concurrency, backup and recovery, rate limiting, and load testing need
explicit verification.

## 9. Recommended Next Sprint

1.  **Strengthen answer-quality evaluation**
    -   Add an annual-profit abstention test.
    -   Test unsupported questions across several domains.
    -   Evaluate generated answers against expected answers and source
        evidence.
2.  **Improve retrieval robustness**
    -   Add a larger and more varied evaluation dataset.
    -   Test paraphrased queries and semantically similar but incorrect
        passages.
    -   Establish score thresholds using measured evaluation results.
3.  **Improve source traceability**
    -   Return useful document and passage references with answers.
    -   Verify that every displayed citation actually supports its
        associated claim.
4.  **Harden document ingestion**
    -   Test malformed files, duplicate uploads, extraction failures,
        and resource limits.
    -   Verify safe handling of untrusted document content.
5.  **Improve observability**
    -   Record ingestion outcomes, retrieval latency, model latency,
        failures, and token usage.
    -   Add structured logs and operational metrics.
6.  **Assess production architecture**
    -   Benchmark concurrent queries and larger document collections.
    -   Review the suitability of SQLite for the intended deployment
        scale.
    -   Establish backup, recovery, and deployment procedures.

## 10. Completion Assessment

Sprint 1 has achieved its principal implementation and validation
objectives for the current RAG foundation.

**Completed and validated:** - Core API endpoints. - Document upload and
validation flows. - Chunking and vector storage. - Hybrid retrieval
ranking. - Context-grounded question answering. - Automated regression
suite. - Initial retrieval evaluation. - Cleanup of the temporary
upload-test record.

**Remaining before a stronger production-readiness claim:** - Profit
abstention validation. - Broader answer-quality evaluation. - Citation
verification. - Security and load testing. - Operational and recovery
checks.

### Final assessment

**Sprint 1 status: Core validation passed; further hardening
recommended.**

The project has a functioning, tested RAG foundation. The next phase
should focus on answer reliability, evidence traceability, ingestion
security, and operational readiness rather than treating the current
evaluation scores as a guarantee of production performance.
