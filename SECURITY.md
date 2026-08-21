# Security Policy

Do not open public issues for credential leaks or security vulnerabilities.
Use GitHub private vulnerability reporting.

Never commit `.env`, Gemini keys, OpenAI keys, database credentials, or production usage data.
If a provider key is committed, rotate it immediately. Removing the line from the newest commit is not sufficient because Git history may still contain the secret.

The `/usage` endpoint returns provider/model, usage, cost, error-code, and request-ID metadata;
it does not store prompts, responses, or provider credentials. It is unauthenticated only when
`GATEWAY_API_KEY` is unset for local demonstration. `APP_ENV=production` requires that key, but
production deployments still need tenant authorization, rate limiting, request-cost limits, TLS,
and a deployment-specific threat model.
