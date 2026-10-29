# Contributing

Thank you for considering a contribution to Enterprise Knowledge Assistant.

## Before you start

1. Search existing issues to avoid duplicating a report.
2. For significant changes, open an issue first to discuss the proposed design.
3. Never include API keys, passwords, confidential documents, or private evaluation labels in a commit.
4. Do not add `tests/data/` or local reviewed relevance labels to Git unless maintainers explicitly approve a safe, public fixture.

## Development setup

```bash
git clone https://github.com/sweetatyagiaz/Enterprise-Knowledge-Assistant.git
cd Enterprise-Knowledge-Assistant
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-app.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Configure a local `.env` file using `.env.example`. Never commit `.env`.

## Tests

Run the full test suite from the repository root:

```bash
PYTHONPATH=. python -m pytest -q
```

For focused tests:

```bash
PYTHONPATH=. python -m pytest tests/test_reviewed_relevance.py -q
```

If a test depends on external services or credentials, document that dependency and keep ordinary unit tests deterministic where possible.

## Pull requests

- Keep changes focused and explain the problem being solved.
- Add or update tests for behavior changes.
- Update documentation and examples when the API or configuration changes.
- Include the commands you ran and their results.
- Do not claim performance or quality improvements without reproducible measurements.
- Preserve the distinction between human-reviewed labels and keyword-based proxy evaluation.
- Keep local data, generated databases, model artifacts, secrets, and reports containing sensitive information out of commits.

## Suggested commit format

Use a short imperative subject, optionally with a category:

- `feat: add ...`
- `fix: handle ...`
- `test: cover ...`
- `docs: explain ...`
- `perf: improve ...`

## Reporting issues

Use the GitHub issue templates where available. Include a minimal reproduction, environment details, expected behavior, actual behavior, and sanitized logs.
