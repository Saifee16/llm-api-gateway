# LLM API Gateway

[![CI](https://github.com/Saifee16/llm-api-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Saifee16/llm-api-gateway/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Saifee16/llm-api-gateway/actions/workflows/codeql.yml/badge.svg)](https://github.com/Saifee16/llm-api-gateway/actions/workflows/codeql.yml)

A provider-neutral FastAPI gateway for routing application requests across multiple LLM providers through stable model aliases.

Instead of coupling an application directly to `gemini-3.5-flash`, `gpt-5.6-terra`, or another provider-specific model name, the application asks for an intent such as `fast`, `reasoning`, `cheap`, or `embedding`.

The gateway owns provider selection, retry policy, fallback routing, timeout enforcement, error normalization, token accounting, cost estimation, request IDs, latency metadata, structured output, embeddings, and usage persistence.

## Why This Repository Exists

A prototype often starts with this:

```python
client = ProviderSDK(...)
response = client.some_api(...)
```

Then model calls spread across route handlers, background jobs, RAG pipelines, agents, and automation code. Provider-specific exceptions leak into application logic. Model names are hard-coded. Cost tracking is inconsistent. Fallback behavior becomes ad hoc.

This project introduces a gateway boundary:

```text
Application
    |
    | model="fast"
    v
LLM API Gateway
    |
    +--> model alias routing
    +--> provider adapter
    +--> timeout
    +--> retry with exponential backoff and jitter
    +--> provider fallback
    +--> normalized errors
    +--> normalized token usage
    +--> cost estimation
    +--> persistent usage events
    |
    v
Gemini / OpenAI
```

## Core Features

- FastAPI API gateway
- Gemini provider adapter
- OpenAI provider adapter
- Provider-neutral internal contracts
- Stable model aliases
- Ordered fallback chains
- Optional provider pinning per request
- Retryable vs non-retryable error classification
- Exponential backoff with jitter
- Gateway-level timeouts
- Request ID propagation
- Response latency headers
- Structured JSON output
- Text embeddings
- Token usage normalization
- Cached-token accounting where provider metadata exposes it
- Configurable pricing catalog
- Estimated request cost
- SQLite usage-event persistence
- Usage summary and provider/model breakdown
- Typed Pydantic responses
- Async endpoint tests
- Real routing/retry/fallback unit tests
- Provider adapter normalization tests
- Ruff
- Pytest coverage gate
- Docker
- Docker Compose persistent SQLite volume
- GitHub Actions CI
- CodeQL
- Dependabot
- Detailed project-specific `NOTES.md`

## Architecture

```text
HTTP Request
    |
    v
RequestContextMiddleware
    |
    +--> X-Request-ID
    +--> latency timer
    |
    v
FastAPI endpoint
    |
    v
GatewayService
    |
    +--> input policy limits
    |
    v
RoutingService
    |
    +--> alias -> ordered provider:model chain
    +--> provider hint filtering
    +--> capability filtering
    +--> configured-provider filtering
    |
    v
Gateway execution loop
    |
    +--> asyncio timeout
    +--> provider adapter
    +--> error classification
    +--> retry transient failure
    +--> exponential backoff + jitter
    +--> fallback to next route
    |
    v
Normalized provider result
    |
    +--> normalized token usage
    +--> price lookup
    +--> estimated cost
    +--> usage event persistence
    |
    v
Typed gateway JSON response
```

## Model Aliases

Default aliases are configured in `.env`:

```env
MODEL_ALIAS_FAST=gemini:gemini-3.5-flash,openai:gpt-5.6-terra
MODEL_ALIAS_REASONING=openai:gpt-5.6-sol,gemini:gemini-3.5-flash
MODEL_ALIAS_CHEAP=gemini:gemini-3.1-flash-lite,openai:gpt-5.6-luna
MODEL_ALIAS_EMBEDDING=gemini:gemini-embedding-001,openai:text-embedding-3-small
```

Each value is an ordered fallback chain.

Example:

```text
fast
 |
 +--> gemini:gemini-3.5-flash
 |
 +--> openai:gpt-5.6-terra
```

If only `GEMINI_API_KEY` is configured, the OpenAI route remains visible in `/models` but is skipped at execution time.

This means the starter works with one provider and becomes multi-provider when a second key is added.

## API Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Process liveness |
| `GET` | `/api/v1/ready` | Database and provider readiness |
| `GET` | `/api/v1/providers` | Provider configuration/capabilities |
| `GET` | `/api/v1/models` | Alias and fallback-chain metadata |
| `POST` | `/api/v1/chat` | Provider-neutral text generation |
| `POST` | `/api/v1/chat/structured` | JSON Schema constrained output |
| `POST` | `/api/v1/embeddings` | Provider-neutral text embeddings |
| `GET` | `/api/v1/usage` | Usage, token, cost, and attempt analytics |

## Quick Start

### 1. Clone

```bash
git clone https://github.com/Saifee16/llm-api-gateway.git
cd llm-api-gateway
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements-dev.txt
```

The requirement files use compatible version ranges around the SDK/tool versions used for final validation. This reduces accidental breakage from future major releases while still allowing compatible updates. Dependabot is configured to propose dependency updates through pull requests.

### 4. Create `.env`

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

macOS/Linux:

```bash
cp .env.example .env
```

Add at least one provider key:

```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=
```

Or:

```env
GEMINI_API_KEY=
OPENAI_API_KEY=your_openai_key
```

For multi-provider fallback, configure both.

`APP_DEBUG` is disabled by default. For any non-local deployment, set `APP_ENV=production`
and a strong `GATEWAY_API_KEY`; the application refuses to start in production without one.
When configured, all routes except liveness and readiness require that value in `X-API-Key`.
Use a secret manager or platform secret injection rather than committing it to `.env`.

### 5. Run

```bash
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Chat Example

```http
POST /api/v1/chat
```

```json
{
  "model": "fast",
  "messages": [
    {
      "role": "system",
      "content": "Explain technical ideas clearly."
    },
    {
      "role": "user",
      "content": "What problem does a provider adapter solve?"
    }
  ]
}
```

Example gateway response shape:

```json
{
  "success": true,
  "request_id": "f7d...",
  "data": {
    "request_id": "f7d...",
    "model_alias": "fast",
    "provider": "gemini",
    "model": "gemini-3.5-flash",
    "content": "...",
    "fallback_used": false,
    "usage": {
      "input_tokens": 42,
      "output_tokens": 81,
      "total_tokens": 123,
      "cached_input_tokens": 0,
      "estimated_cost_usd": 0.000792
    },
    "attempts": [
      {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "attempt_number": 1,
        "status": "success",
        "latency_ms": 842.1,
        "error_code": null
      }
    ]
  }
}
```

## Force a Configured Provider

Normally the alias policy chooses the route.

For debugging or evaluation, a request may include:

```json
{
  "model": "fast",
  "provider": "openai",
  "messages": [
    {
      "role": "user",
      "content": "Explain RAG in one sentence."
    }
  ]
}
```

The provider hint does not bypass model aliases. It filters the alias chain.

Therefore the application still asks for `fast`; it does not send a raw provider model name through the public API.

## Structured Output

```http
POST /api/v1/chat/structured
```

```json
{
  "model": "fast",
  "messages": [
    {
      "role": "user",
      "content": "Extract a title and difficulty from: FastAPI dependency injection is useful."
    }
  ],
  "schema_name": "lesson",
  "response_schema": {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "difficulty": {
        "type": "string",
        "enum": ["beginner", "intermediate", "advanced"]
      }
    },
    "required": ["title", "difficulty"],
    "additionalProperties": false
  }
}
```

The provider adapters translate the same internal schema request into provider-specific SDK arguments.

## Embeddings

```http
POST /api/v1/embeddings
```

```json
{
  "model": "embedding",
  "inputs": [
    "FastAPI builds Python APIs.",
    "Embeddings map content into vectors."
  ]
}
```

Response metadata includes:

```text
provider
model
dimensions
normalized token usage
estimated cost when the pricing catalog contains a rate
attempt history
```

## Retry and Fallback Policy

The gateway classifies provider errors into:

```text
rate_limit
transient
auth
invalid_request
not_supported
configuration
unknown
```

Retry policy:

```text
429 / rate limit       -> retry
408 / timeout          -> retry
5xx transient failure  -> retry
invalid request        -> fail fast
```

After retry attempts are exhausted, the gateway moves to the next configured route in the alias chain.

Backoff:

```text
base * 2^retry_index
```

with bounded random jitter.

Example:

```text
0.25s + jitter
0.50s + jitter
1.00s + jitter
```

The delay is capped by `GATEWAY_BACKOFF_MAX_SECONDS`.

## Usage and Cost Accounting

Every provider attempt is persisted.

This includes failed attempts.

A usage row records:

```text
request_id
operation
model_alias
provider
model
attempt_number
success
error_code
latency_ms
input_tokens
output_tokens
total_tokens
cached_input_tokens
estimated_cost_usd
created_at
```

Why failed attempts are stored:

```text
logical request
    |
    +--> Gemini 429
    +--> Gemini 429
    +--> OpenAI success
```

If only the final OpenAI response were stored, the gateway would hide the operational reason that fallback occurred.

Query:

```http
GET /api/v1/usage?days=7&limit=100
```

The response contains:

```text
total attempts
successful attempts
failed attempts
input tokens
output tokens
total tokens
estimated cost
provider/model breakdown
recent attempt events
```

## Pricing Catalog

Pricing lives in:

```text
config/pricing.json
```

It is deliberately outside provider adapters.

Provider adapters answer:

> How many tokens did the provider report?

The pricing catalog answers:

> What is the configured estimated price for those tokens?

The catalog included in this repository is a dated snapshot, not a billing authority. Update it when provider prices change.

Models with unknown configured prices return:

```json
{
  "estimated_cost_usd": null
}
```

The gateway does not silently invent a price.

## Request IDs

Clients may send:

```http
X-Request-ID: my-trace-id
```

If absent, the gateway creates a UUID.

The ID appears in:

```text
response header
success body
error body
usage-event table
application logs
```

The response also includes:

```http
X-Response-Time-Ms
```

This gives a simple correlation mechanism before introducing a full observability platform.

## Tests

Run:

```bash
pytest
```

Coverage gate:

```bash
pytest --cov=app --cov-report=term-missing --cov-fail-under=80
```

The current suite verifies:

- health and readiness
- request-ID propagation
- provider metadata
- alias metadata
- pricing
- ordered routing
- provider hints
- unknown aliases
- retry-before-fallback behavior
- fallback after retries fail
- fail-fast invalid requests
- direct structured gateway execution
- direct embedding gateway execution
- Gemini result normalization with fake SDK responses
- OpenAI result normalization with fake SDK responses
- provider configuration errors
- chat endpoint
- structured endpoint
- embedding endpoint
- JSON Schema request validation
- usage aggregation
- empty usage endpoint

The tests do not make billable provider calls.

## Code Quality

```bash
ruff check .
ruff format --check .
bandit -q -r app
pip-audit -r requirements.txt
```

Format:

```bash
ruff format .
```

## Docker

Build:

```bash
docker build -t llm-api-gateway .
```

Run:

```bash
docker run --env-file .env -p 8000:8000 llm-api-gateway
```

## Docker Compose

```bash
docker compose up --build
```

Compose binds to `127.0.0.1` by default. Set `HOST_PORT` to use a different local port;
put a reverse proxy with TLS, authentication, authorization, and rate limiting in front of any
internet-facing deployment.

Compose mounts a named volume at `/app/data` and overrides the SQLite URL:

```text
sqlite:////app/data/gateway.db
```

Therefore usage records survive container replacement.

Stop:

```bash
docker compose down
```

Delete the persistent database volume too:

```bash
docker compose down -v
```

## CI

GitHub Actions runs three independent jobs:

```text
quality
  -> Ruff lint
  -> Ruff format check

test
  -> pytest
  -> coverage XML
  -> 80% floor

docker
  -> Docker image build
```

CodeQL runs on pushes, pull requests, and a weekly schedule.

Dependabot monitors Python, Docker, and GitHub Actions dependencies.

## Scope Boundaries

This repository intentionally does not include:

- JWT authentication
- per-tenant API keys
- Redis distributed rate limiting
- streaming responses
- tool/function execution
- prompt caching
- semantic response caching
- circuit breakers shared across replicas
- OpenTelemetry
- Prometheus
- Kubernetes

Those are valid future extensions, but adding all of them would hide the project's core lesson: provider abstraction and reliable LLM request routing.

## Security Warning

The `/usage` route is unauthenticated only when no `GATEWAY_API_KEY` is configured, which is
intended for local portfolio demonstration. It exposes provider/model, usage, cost, error-code,
and request-ID metadata (but never prompts or provider credentials).

Do not expose this gateway directly to the public internet without authentication, authorization,
rate limiting, and a deployment-specific threat model. `APP_ENV=production` requires gateway-key
authentication, but tenant authorization, distributed rate limiting, and abuse/cost controls remain
deployment responsibilities.

See `SECURITY.md`.

## Detailed Learning Notes

Read `NOTES.md`.

The notes assume you already completed:

1. AI Engineering Starter Kit
2. Lead Scoring ML API

They focus on concepts that are new in Project 3.

## License

MIT. See `LICENSE`.
