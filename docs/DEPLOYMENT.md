# Deployment and operations notes

## Intended use

The current project is a local learning/portfolio prototype using FastAPI, OpenAI model APIs, SQLite, and NumPy-based retrieval. Do not expose it to the public internet or use it for sensitive workloads without a separate security and operations review.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m uvicorn api.main:app --reload
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

Check:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

Use `.env.example` to prepare local configuration. The actual environment-variable names and defaults must match `ai_core/settings.py`.

## Data and secrets

- Keep `.env` out of version control.
- Keep local databases, uploaded documents, generated embeddings, and evaluation reports out of commits unless specifically reviewed for public release.
- Use synthetic or public documents for demonstrations.
- Review external model-provider data handling before sending any sensitive content.
- Document how local data can be deleted and whether deletion also removes derived chunks or embeddings.

## Before a production deployment

The following are review requirements, not claims about current implementation:

- Authentication and authorization
- Tenant isolation and document-level access controls
- Rate limiting and abuse protection
- Upload validation, malware scanning, and resource limits
- Secret management and credential rotation
- TLS and secure reverse-proxy configuration
- Database backup, recovery, and retention policy
- Logging that avoids secrets and sensitive document contents
- Monitoring, alerting, dependency updates, and incident response
- Load, latency, and memory testing with representative workloads
- Prompt-injection and untrusted-document threat assessment
- Privacy/legal review for the intended jurisdiction and data type

## Containerization

Docker support is a roadmap item unless a maintained Dockerfile and compose configuration are present in the repository. Do not assume this bundle provides a production container image.
