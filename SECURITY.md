# Security Policy

## Supported versions

This project is currently a learning and portfolio project. Security fixes are assessed against the latest version on the default branch. No formal service-level agreement or guaranteed response time is offered.

## Reporting a vulnerability

Please do not publish exploitable vulnerability details, credentials, personal data, or private documents in a public issue.

Until a private security contact is configured for this repository, contact the maintainers through a trusted private channel available from the repository owner profile and ask for a private vulnerability-reporting route. Do not include secrets in the initial message.

A useful report includes:

- A concise description and potential impact
- Affected commit or version
- Steps to reproduce in a safe test environment
- Relevant sanitized logs or proof of concept
- Any suggested mitigation

Please allow maintainers reasonable time to investigate before public disclosure.

## Security expectations and known limitations

- Never commit `.env`, API keys, passwords, tokens, or production credentials.
- Do not upload confidential documents to a test instance unless you have authorization and have reviewed the model provider's data handling.
- Treat model output as untrusted content and validate it before using it in downstream systems.
- Review authentication, authorization, rate limiting, tenant isolation, upload scanning, retention, and deployment hardening before exposing an instance to the public internet.
- This policy does not imply that all of these controls are implemented in the current application.
