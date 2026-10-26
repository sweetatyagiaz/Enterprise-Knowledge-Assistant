# Enterprise Knowledge Assistant

A starter Retrieval-Augmented Generation (RAG) application built with
**FastAPI**, **OpenAI embeddings**, **SQLite**, and **NumPy**. Upload
documents, index their text chunks, and ask questions grounded in
retrieved passages.

## Features

-   REST API documented with Swagger UI.
-   Document upload for PDF, DOCX, TXT, and Markdown files.
-   Text chunking with configurable overlap.
-   Embedding generation using the OpenAI Embeddings API.
-   Local SQLite storage for document chunks and embedding vectors.
-   Cosine-similarity retrieval.
-   Question answering with source information.
-   Basic health and document-listing endpoints.

> This is a learning and portfolio project. The current SQLite/NumPy
> vector search is intended for a small local prototype, not a
> high-scale production deployment.

## Architecture

``` text
Document (PDF/DOCX/TXT/MD)
          |
          v
  Document Loader
          |
          v
    Text Chunking
          |
          v
  OpenAI Embeddings
          |
          v
 SQLite Vector Store
          |
          +----------------------+
                                 |
Question -> Query Embedding -----+
                                 |
                                 v
                      Similarity Search
                                 |
                                 v
                       Retrieved Context
                                 |
                                 v
                       OpenAI Chat Model
                                 |
                                 v
                         Answer + Sources
```

## Project structure

``` text
ai-architect-portfolio/
├── ai_core/
│   ├── __init__.py
│   ├── settings.py
│   ├── chunking.py
│   ├── rag.py
│   ├── vector_store.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── service.py
│   └── ingestion/
│       ├── __init__.py
│       └── loaders.py
├── api/
│   └── main.py
├── tests/
├── data/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Your local folder structure may contain additional files as the project
develops.


## Main RAG Architecture Components

1. **Document Loader:** Extracts text from PDF, DOCX, TXT, and Markdown documents.
2. **Text Chunking:** Splits document text into smaller, overlapping chunks.
3. **OpenAI Embeddings:** Uses the `text-embedding-3-small` embedding model by default to convert text into 1536-dimensional vectors.
4. **SQLite Vector Store:** Stores document chunks and their embedding vectors in `data/vector_store.sqlite3`.
5. **Similarity Search:** Uses cosine similarity to retrieve the most relevant document chunks for a user question.
6. **Retrieved Context:** Combines retrieved passages to provide context for answering the question.
7. **OpenAI Chat Model:** Uses `gpt-4.1-mini` by default to generate answers grounded in the retrieved context.
8. **FastAPI:** Exposes endpoints for document ingestion, document listing, health checks, and question answering.

### Default Retrieval Configuration

The default number of retrieved chunks is 5 per question (`TOP_K=5`). The application can use a different number when the request explicitly supplies `top_k`.


## Requirements

-   Python 3.10 or newer
-   An OpenAI API key with API access and available billing/credits
-   macOS, Linux, or Windows

## Setup

Run these commands from the project root.

### 1. Activate your virtual environment

If your environment is already activated, skip this step. For example:

``` bash
source venv_django/bin/activate
```

Use the activation command that matches the name and location of your
own virtual environment.

### 2. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

If `requirements.txt` is not complete yet, install the required
packages:

``` bash
python -m pip install fastapi "uvicorn[standard]" pydantic-settings openai pypdf python-docx numpy python-multipart pytest httpx
```

## Default AI Models

The default embedding model is `text-embedding-3-small`, configured through `EMBEDDING_MODEL`. It generates 1536-dimensional embedding vectors for document chunks and user questions.

The default chat completion model is `gpt-4.1-mini`, configured through `CHAT_MODEL`. It generates answers using the context retrieved from the SQLite vector store.

The project stores document chunks and embedding vectors in SQLite at `data/vector_store.sqlite3`. Retrieval uses cosine similarity and returns 5 chunks by default (`top_k=5`).


### 3. Configure environment variables

Create a local `.env` file:

``` bash
cp .env.example .env
```

If `.env.example` does not exist, create `.env` manually with:

``` dotenv
OPENAI_API_KEY=replace_with_your_openai_api_key
CHAT_MODEL=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=1536
DATA_DIR=data
SQLITE_PATH=data/vector_store.sqlite3
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=5
MAX_UPLOAD_MB=15
```

Replace the placeholder with your real API key. **Never commit `.env` or
expose your API key in source control.**

## Run the API

From the project root:

``` bash
python -m uvicorn api.main:app --reload
```

Open Swagger UI at http://127.0.0.1:8000/docs.

Health endpoint: http://127.0.0.1:8000/health

## Try it

### 1. Upload a document

Using Swagger UI:

1.  Open `/docs`.
2.  Expand `POST /documents`.
3.  Click **Try it out**.
4.  Choose a PDF, DOCX, TXT, or Markdown file.
5.  Click **Execute**.

Or upload this README from the project root using `curl`:

``` bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@README.md"
```

A successful response should include the filename, number of chunks
created, and indexing status.

### 2. Verify indexed documents

Use `GET /documents` in Swagger UI, or run:

``` bash
curl http://127.0.0.1:8000/documents
```

### 3. Ask a question

In Swagger UI, expand `POST /ask` and submit:

``` json
{
  "question": "What are the main components of this project?",
  "top_k": 5
}
```

Or use `curl`:

``` bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the main components of this project?","top_k":5}'
```

The API should return an answer and source information from retrieved
chunks. Actual responses depend on the uploaded document and successful
OpenAI API access.

## API endpoints

  Method   Endpoint       Purpose
  -------- -------------- ----------------------------------------
  `GET`    `/health`      Check API status
  `GET`    `/documents`   List indexed documents and chunk count
  `POST`   `/documents`   Upload and index a document
  `POST`   `/ask`         Ask a question using retrieved context

## Configuration

  ------------------------------------------------------------------------------
  Variable                 Purpose                 Default
  ------------------------ ----------------------- -----------------------------
  `OPENAI_API_KEY`         OpenAI API              Required for model calls
                           authentication          

  `CHAT_MODEL`             Chat completion model   `gpt-4.1-mini`

  `EMBEDDING_MODEL`        Embedding model         `text-embedding-3-small`

  `EMBEDDING_DIMENSIONS`   Embedding vector        `1536`
                           dimensions              

  `SQLITE_PATH`            SQLite database         `data/vector_store.sqlite3`
                           location                

  `CHUNK_SIZE`             Characters per text     `900`
                           chunk                   

  `CHUNK_OVERLAP`          Overlap between chunks  `150`

  `TOP_K`                  Number of retrieved     `5`
                           chunks                  

  `MAX_UPLOAD_MB`          Maximum upload size in  `15`
                           MB                      
  ------------------------------------------------------------------------------

The environment variable names and defaults should match
`ai_core/settings.py`.

## Development checks

Check imports:

``` bash
python -c "from ai_core.rag import KnowledgeAssistant; print('RAG import successful')"
```

Run tests, if the test suite is present:

``` bash
python -m pytest -q
```

## Current limitations and next steps

Before using this as a production service, add and test:

-   Authentication, authorization, and per-document access controls.
-   File-content validation and robust upload limits.
-   Better PDF extraction, OCR for scanned documents, and handling of
    tables.
-   Automated evaluation for retrieval relevance and answer grounding.
-   Structured logging, latency metrics, tracing, and cost monitoring.
-   Rate limiting, timeouts, retries, and safe error handling.
-   A scalable vector database/index if the corpus or traffic grows.
-   Deployment configuration, CI, and security tests.

## Learning objectives

This project demonstrates the core RAG flow:

1.  Ingest documents.
2.  Split text into overlapping chunks.
3.  Generate embeddings.
4.  Store and retrieve vectors.
5.  Build context from retrieved passages.
6.  Generate an answer grounded in that context.
7.  Evaluate retrieval quality, answer faithfulness, latency, and cost.

------------------------------------------------------------------------

**Project:** AI Architect Portfolio --- Sprint 1\
**Component:** Enterprise Knowledge Assistant
