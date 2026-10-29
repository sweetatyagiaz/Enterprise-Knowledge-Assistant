# API reference and examples

The examples below reflect the routes documented in the project README. Use the running application's OpenAPI page at `http://127.0.0.1:8000/docs` as the authoritative source for the exact request fields, validation constraints, and response schema.

## Run the API

```bash
python -m uvicorn api.main:app --reload
```

Base URL: `http://127.0.0.1:8000`

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check API health |
| `GET` | `/documents` | List indexed documents and chunk counts |
| `POST` | `/documents` | Upload and index a document |
| `POST` | `/ask` | Ask a question using retrieved context |

## Health check

```bash
curl -i http://127.0.0.1:8000/health
```

## Upload a document

```bash
curl -X POST "http://127.0.0.1:8000/documents" \
  -F "file=@README.md"
```

Supported document formats documented by the project: PDF, DOCX, TXT, and Markdown. Upload-size and content validation are enforced by the application; check settings and API responses for the current limits.

## List documents

```bash
curl -i http://127.0.0.1:8000/documents
```

## Ask a question

```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main components of this project?"
  }'
```

The response should contain an answer and source information according to the current API implementation. Do not rely on a guessed response schema: inspect `/docs` or the OpenAPI JSON for the running version.

## Troubleshooting

- **Connection refused:** Check that Uvicorn is running and that the URL/port match.
- **Validation error:** Compare the submitted body and fields with `/docs`.
- **Upload rejected:** Check supported file types, upload-size settings, and whether the file contains extractable text.
- **Embedding or model API error:** Check network access, provider status, configured model names, API key validity, and available account access. Never paste the key into an issue.
- **No useful answer:** Verify that the document was indexed, that the question is answerable from the indexed text, and that the returned sources contain the relevant evidence.
