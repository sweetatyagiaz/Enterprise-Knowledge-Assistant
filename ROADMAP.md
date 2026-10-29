# Roadmap

This roadmap is a planning document, not a statement that listed features are already implemented. Priorities may change based on feedback, security review, and project scope.

## Completed foundations

- [x] FastAPI application for document ingestion and question answering
- [x] PDF, DOCX, TXT, and Markdown ingestion paths
- [x] Chunking, embeddings, local SQLite storage, and similarity retrieval
- [x] Request validation and ingestion error-path tests
- [x] Grounding and unsupported-answer evaluation checks
- [x] Retrieval ranking metrics: Precision@K, Recall@K, MRR, and nDCG@K
- [x] Structured evaluation report output
- [x] Human-reviewed chunk-level relevance evaluation workflow
- [x] Vector search/cache performance work documented in the sprint history

## Near-term priorities

- [ ] Verify clean-machine installation instructions and keep `.env.example` synchronized with application settings
- [ ] Add real screenshots and a short, sanitized end-to-end demo
- [ ] Document every API request/response schema from the running OpenAPI specification
- [ ] Add CI and publish its actual test status
- [ ] Add public, non-sensitive evaluation fixtures with stable chunk identifiers or a reproducible relabeling process
- [ ] Document data retention, deletion, and external model-provider data flow
- [ ] Select and add an appropriate open-source license

## Medium-term possibilities

- [ ] Authentication and authorization for deployed environments
- [ ] Rate limiting and upload abuse protections
- [ ] Improved observability and structured request logging
- [ ] Docker-based development and deployment
- [ ] Better extraction for scanned PDFs, potentially including OCR
- [ ] Configurable retrieval strategies and systematic ablation studies
- [ ] Larger, representative relevance evaluation set with documented annotation guidelines
- [ ] Deployment-specific database and vector-store options when justified by measured needs

## Longer-term possibilities

- [ ] Multi-user and tenant-aware data isolation
- [ ] Background ingestion for larger documents and collections
- [ ] Evaluation regression tracking across code/model/configuration changes
- [ ] Deployment security review and load testing
- [ ] User feedback loop for reviewing answer quality and source usefulness

## How priorities are chosen

Prioritize correctness, reproducibility, security, and clear documentation before adding complexity. Any claimed improvement should include a reproducible test or benchmark and its limitations.
