# Project 3 Notes — LLM API Gateway

These notes are written for Saifee after Project 1 (`ai-engineering-starter-kit`) and Project 2 (`lead-scoring-ml-api`). They intentionally do not reteach basic FastAPI routing, Pydantic, `.env`, Docker, pytest, Ruff, or the general service-layer idea unless Project 3 changes how those concepts are used.

The goal is to understand the *new engineering boundary*: an application should not need provider-specific code everywhere it calls an LLM. Project 3 creates a provider-neutral gateway that owns reliability, routing, accounting, and response normalization.

## 0. What is new compared with Project 1 and Project 2?

```text
Project 1
  FastAPI backend engineering

Project 2
  offline ML training -> persisted model -> online inference

Project 3
  application request -> model intent -> provider routing -> reliability policy -> normalized response
```

In Project 1, `GeminiService` directly knew Gemini. That was correct for a starter kit because the lesson was layering. Project 3 asks a harder architecture question: what happens when ten routes, agents, RAG systems, and background jobs need LLM calls and you may change providers?

The answer is not to duplicate `if provider == "gemini"` throughout the application. The answer is to isolate provider-specific behavior behind adapters and keep routing policy outside those adapters.

## 1. Complete repository structure

```text
llm-api-gateway/
├── app/
│   ├── api/v1/endpoints/
│   │   ├── gateway.py
│   │   ├── health.py
│   │   ├── metadata.py
│   │   └── usage.py
│   ├── core/
│   │   ├── config.py
│   │   ├── exceptions.py
│   │   ├── logging.py
│   │   ├── middleware.py
│   │   └── pricing.py
│   ├── db/
│   │   ├── base.py
│   │   ├── database.py
│   │   └── models.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── exceptions.py
│   │   ├── gemini.py
│   │   ├── openai.py
│   │   └── registry.py
│   ├── repositories/
│   │   └── usage_repository.py
│   ├── schemas/
│   │   ├── common.py
│   │   ├── gateway.py
│   │   ├── health.py
│   │   ├── provider.py
│   │   └── usage.py
│   ├── services/
│   │   ├── gateway_service.py
│   │   ├── routing_service.py
│   │   └── usage_service.py
│   └── main.py
├── config/
│   └── pricing.json
├── tests/
│   ├── fakes.py
│   ├── test_gateway_endpoints.py
│   ├── test_gateway_service.py
│   ├── test_health.py
│   ├── test_metadata.py
│   ├── test_pricing.py
│   ├── test_providers.py
│   ├── test_routing.py
│   └── test_usage.py
├── .github/
├── .env.example
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── README.md
└── NOTES.md
```

The biggest new folders are `providers/` and `repositories/`. `providers/` isolates third-party SDK behavior. `repositories/` isolates SQL queries from business logic. This project therefore adds two boundaries that did not exist as real working boundaries in Project 1.

## 2. Mental model: intent alias vs provider model

```text
Client says:        model = "fast"
Gateway policy says: fast -> Gemini first -> OpenAI second
Provider adapter says: translate normalized request to provider SDK
```

`fast` is a business/application intent. `gemini-3.5-flash` is a provider model identifier. Mixing those concepts means the application becomes coupled to vendor product names.

A gateway creates indirection. Indirection means one name points to another layer of decision-making. It adds complexity, so only use it when the flexibility is useful. In this project the whole lesson is exactly that flexibility.

## 3. Configuration: model aliases are policy

**File: `app/core/config.py`**

```python
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="LLM API Gateway", alias="APP_NAME")
    app_env: Literal["local", "development", "staging", "production"] = Field(
        default="local",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    database_url: str = Field(default="sqlite:///./gateway.db", alias="DATABASE_URL")
    pricing_catalog_path: Path = Field(
        default=Path("config/pricing.json"),
        alias="PRICING_CATALOG_PATH",
    )

    gateway_timeout_seconds: float = Field(default=45.0, alias="GATEWAY_TIMEOUT_SECONDS")
    gateway_max_retries: int = Field(default=2, ge=0, le=8, alias="GATEWAY_MAX_RETRIES")
    gateway_backoff_base_seconds: float = Field(
        default=0.25,
        ge=0.0,
        alias="GATEWAY_BACKOFF_BASE_SECONDS",
    )
    gateway_backoff_max_seconds: float = Field(
        default=4.0,
        ge=0.0,
        alias="GATEWAY_BACKOFF_MAX_SECONDS",
    )

    model_alias_fast: str = Field(
        default="gemini:gemini-3.5-flash,openai:gpt-5.6-terra",
        alias="MODEL_ALIAS_FAST",
    )
    model_alias_reasoning: str = Field(
        default="openai:gpt-5.6-sol,gemini:gemini-3.5-flash",
        alias="MODEL_ALIAS_REASONING",
    )
    model_alias_cheap: str = Field(
        default="gemini:gemini-3.1-flash-lite,openai:gpt-5.6-luna",
        alias="MODEL_ALIAS_CHEAP",
    )
    model_alias_embedding: str = Field(
        default="gemini:gemini-embedding-001,openai:text-embedding-3-small",
        alias="MODEL_ALIAS_EMBEDDING",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    max_messages: int = Field(default=50, ge=1, le=500, alias="MAX_MESSAGES")
    max_message_chars: int = Field(default=20_000, ge=100, alias="MAX_MESSAGE_CHARS")
    max_embedding_inputs: int = Field(default=100, ge=1, alias="MAX_EMBEDDING_INPUTS")
    max_embedding_chars: int = Field(default=8_000, ge=100, alias="MAX_EMBEDDING_CHARS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

The important new fields are the four `model_alias_*` strings. A chain uses `provider:model` entries separated by commas. The order is meaningful. The first usable target is preferred; later targets are fallbacks.

```env
MODEL_ALIAS_FAST=gemini:gemini-3.5-flash,openai:gpt-5.6-terra
```

Read this from left to right: for the `fast` intent, prefer Gemini 3.5 Flash. If that route cannot complete after the gateway retry policy, try OpenAI GPT-5.4 mini. If the OpenAI key is missing, the route is visible as metadata but filtered out of executable targets.

Why configuration instead of Python code? Because changing route policy should not require editing endpoint functions. This is a simple form of operational configuration.

## 4. Provider capability contracts

**File: `app/providers/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderCapability(StrEnum):
    CHAT = "chat"
    STRUCTURED = "structured"
    EMBEDDINGS = "embeddings"


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class ProviderChatInput:
    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ProviderStructuredInput(ProviderChatInput):
    schema_name: str = "response"
    response_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderEmbeddingInput:
    model: str
    inputs: list[str]


@dataclass(frozen=True)
class ProviderChatResult:
    content: str
    usage: ProviderUsage
    provider_request_id: str | None = None
    structured_data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderEmbeddingResult:
    embeddings: list[list[float]]
    usage: ProviderUsage
    provider_request_id: str | None = None


class LLMProvider(ABC):
    name: str
    capabilities: frozenset[ProviderCapability]

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities

    @abstractmethod
    async def chat(self, request: ProviderChatInput) -> ProviderChatResult:
        raise NotImplementedError

    @abstractmethod
    async def structured(self, request: ProviderStructuredInput) -> ProviderChatResult:
        raise NotImplementedError

    @abstractmethod
    async def embed(self, request: ProviderEmbeddingInput) -> ProviderEmbeddingResult:
        raise NotImplementedError
```

This file is the heart of the provider abstraction. `LLMProvider` is an abstract base class. An abstract base class defines the methods concrete providers must implement.

```python
class LLMProvider(ABC):
    @abstractmethod
    async def chat(...) -> ProviderChatResult:
        ...

    @abstractmethod
    async def structured(...) -> ProviderChatResult:
        ...

    @abstractmethod
    async def embed(...) -> ProviderEmbeddingResult:
        ...
```

The gateway service calls these methods. It never calls `client.responses.create` and never calls `client.interactions.create`. That provider-specific knowledge is below the interface.

The dataclasses are normalized internal DTOs. DTO means Data Transfer Object: a small object whose job is to move structured data between layers.

```python
ProviderUsage(
    input_tokens=...,
    output_tokens=...,
    total_tokens=...,
    cached_input_tokens=...,
)
```

Both providers report usage differently. The gateway does not want two different response shapes. Each adapter converts provider metadata into `ProviderUsage`. This is normalization.

## 5. Why capabilities exist

```python
class ProviderCapability(StrEnum):
    CHAT = "chat"
    STRUCTURED = "structured"
    EMBEDDINGS = "embeddings"
```

A route is not valid only because the provider exists. The provider must also support the requested operation. Routing therefore filters by both configuration and capability.

```python
provider.configured and provider.supports(capability)
```

This prevents the routing layer from sending an embeddings request to an adapter that only supports chat.

## 6. Normalized provider errors

**File: `app/providers/exceptions.py`**

```python
from enum import StrEnum


class ProviderErrorKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    TRANSIENT = "transient"
    AUTH = "auth"
    INVALID_REQUEST = "invalid_request"
    NOT_SUPPORTED = "not_supported"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ProviderError(Exception):
    def __init__(
        self,
        *,
        provider: str,
        message: str,
        code: str,
        kind: ProviderErrorKind,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.message = message
        self.code = code
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(message)
```

A provider SDK may raise `RateLimitError`, an HTTP exception, a gRPC-style error, or another SDK-specific type. If those SDK exceptions escape into `GatewayService`, the gateway becomes provider-aware again.

The adapters convert errors into `ProviderError`. The two most important fields are `kind` and `retryable`. `kind` explains the class of failure. `retryable` answers the operational policy question: should the same route be tried again?

```text
429                 -> RATE_LIMIT, retryable=True
408 / 5xx / timeout -> TRANSIENT, retryable=True
401 / 403           -> AUTH, retryable=False
400 / 422           -> INVALID_REQUEST, retryable=False
```

Do not retry a bad API key ten times. Do not retry malformed JSON ten times. Retries are for transient failures.

## 7. Gemini adapter

**File: `app/providers/gemini.py`**

```python
import json
from typing import Any

from anyio import to_thread

from app.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderChatInput,
    ProviderChatResult,
    ProviderEmbeddingInput,
    ProviderEmbeddingResult,
    ProviderStructuredInput,
    ProviderUsage,
)
from app.providers.exceptions import ProviderError, ProviderErrorKind


class GeminiProvider(LLMProvider):
    name = "gemini"
    capabilities = frozenset({
        ProviderCapability.CHAT,
        ProviderCapability.STRUCTURED,
        ProviderCapability.EMBEDDINGS,
    })

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._client: Any | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> Any:
        if not self.configured:
            raise ProviderError(
                provider=self.name,
                message="Gemini API key is missing.",
                code="GEMINI_API_KEY_MISSING",
                kind=ProviderErrorKind.CONFIGURATION,
                retryable=False,
            )
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _conversation(messages: list[dict[str, str]]) -> tuple[str | None, str]:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        conversation = [
            f"{m['role'].upper()}: {m['content']}" for m in messages if m["role"] != "system"
        ]
        return "\n\n".join(system_parts) or None, "\n\n".join(conversation)

    @staticmethod
    def _extract_usage(response: Any) -> ProviderUsage:
        usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
        if usage is None:
            return ProviderUsage()

        def first(*names: str) -> int:
            for name in names:
                value = getattr(usage, name, None)
                if value is not None:
                    return int(value)
            return 0

        input_tokens = first("input_tokens", "prompt_token_count")
        output_tokens = first("output_tokens", "candidates_token_count")
        total_tokens = first("total_tokens", "total_token_count") or input_tokens + output_tokens
        cached_tokens = first("cached_input_tokens", "cached_content_token_count")
        return ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
        )

    def _classify(self, exc: Exception) -> ProviderError:
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        try:
            status_int = int(status_code) if status_code is not None else None
        except (TypeError, ValueError):
            status_int = None

        if status_int == 429:
            kind, retryable = ProviderErrorKind.RATE_LIMIT, True
        elif status_int in {408, 500, 502, 503, 504}:
            kind, retryable = ProviderErrorKind.TRANSIENT, True
        elif status_int in {401, 403}:
            kind, retryable = ProviderErrorKind.AUTH, False
        elif status_int in {400, 404, 422}:
            kind, retryable = ProviderErrorKind.INVALID_REQUEST, False
        else:
            kind, retryable = ProviderErrorKind.UNKNOWN, False

        return ProviderError(
            provider=self.name,
            message=str(exc),
            code=f"GEMINI_{kind.value.upper()}",
            kind=kind,
            retryable=retryable,
            status_code=status_int,
        )

    async def chat(self, request: ProviderChatInput) -> ProviderChatResult:
        client = self._get_client()
        system_instruction, transcript = self._conversation(request.messages)
        generation_config: dict[str, object] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            generation_config["max_output_tokens"] = request.max_output_tokens

        def call() -> Any:
            kwargs: dict[str, object] = {
                "model": request.model,
                "input": transcript,
            }
            if system_instruction:
                kwargs["system_instruction"] = system_instruction
            if generation_config:
                kwargs["generation_config"] = generation_config
            return client.interactions.create(**kwargs)

        try:
            response = await to_thread.run_sync(call)
            return ProviderChatResult(
                content=response.output_text,
                usage=self._extract_usage(response),
                provider_request_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc

    async def structured(self, request: ProviderStructuredInput) -> ProviderChatResult:
        client = self._get_client()
        system_instruction, transcript = self._conversation(request.messages)
        generation_config: dict[str, object] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            generation_config["max_output_tokens"] = request.max_output_tokens

        def call() -> Any:
            kwargs: dict[str, object] = {
                "model": request.model,
                "input": transcript,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": request.response_schema,
                },
            }
            if system_instruction:
                kwargs["system_instruction"] = system_instruction
            if generation_config:
                kwargs["generation_config"] = generation_config
            return client.interactions.create(**kwargs)

        try:
            response = await to_thread.run_sync(call)
            content = response.output_text
            return ProviderChatResult(
                content=content,
                structured_data=json.loads(content),
                usage=self._extract_usage(response),
                provider_request_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc

    async def embed(self, request: ProviderEmbeddingInput) -> ProviderEmbeddingResult:
        client = self._get_client()

        def call() -> Any:
            return client.models.embed_content(model=request.model, contents=request.inputs)

        try:
            response = await to_thread.run_sync(call)
            embeddings = [list(item.values) for item in response.embeddings]
            return ProviderEmbeddingResult(
                embeddings=embeddings,
                usage=self._extract_usage(response),
                provider_request_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc
```

The adapter lazy-loads the SDK client. `self._client` begins as `None`; `_get_client()` creates the Google GenAI client only when the provider is actually used.

```python
if self._client is None:
    from google import genai

    self._client = genai.Client(api_key=self.api_key)
```

Lazy creation is useful in a multi-provider process. An application configured only for OpenAI does not need to instantiate a Gemini client at startup.

`_conversation()` currently converts normalized gateway messages into a system instruction plus a transcript string. This is an explicit adapter choice. The public API remains role-based, but the Gemini Interactions call receives the normalized transcript. If the Gemini SDK conversation-input contract evolves, only this adapter changes.

The chat method is asynchronous from the gateway's point of view, but the Interactions SDK call used here is synchronous. `anyio.to_thread.run_sync` moves the blocking SDK call into a worker thread so it does not block the event loop.

```python
response = await to_thread.run_sync(call)
```

That line is new and important. `async def` does not magically make blocking code non-blocking. A synchronous network call inside `async def` can still freeze the event loop.

Structured output translates the gateway schema into Gemini's `response_format` shape. The adapter then parses `output_text` with `json.loads` and returns normalized `structured_data`.

## 8. OpenAI adapter

**File: `app/providers/openai.py`**

```python
import json
from typing import Any

from app.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderChatInput,
    ProviderChatResult,
    ProviderEmbeddingInput,
    ProviderEmbeddingResult,
    ProviderStructuredInput,
    ProviderUsage,
)
from app.providers.exceptions import ProviderError, ProviderErrorKind


class OpenAIProvider(LLMProvider):
    name = "openai"
    capabilities = frozenset({
        ProviderCapability.CHAT,
        ProviderCapability.STRUCTURED,
        ProviderCapability.EMBEDDINGS,
    })

    def __init__(self, api_key: str, timeout_seconds: float) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._client: Any | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self) -> Any:
        if not self.configured:
            raise ProviderError(
                provider=self.name,
                message="OpenAI API key is missing.",
                code="OPENAI_API_KEY_MISSING",
                kind=ProviderErrorKind.CONFIGURATION,
                retryable=False,
            )
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
        return self._client

    @staticmethod
    def _usage(response: Any) -> ProviderUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return ProviderUsage()
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or input_tokens + output_tokens)
        details = getattr(usage, "input_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        return ProviderUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached_tokens,
        )

    def _classify(self, exc: Exception) -> ProviderError:
        status_code = getattr(exc, "status_code", None)
        name = exc.__class__.__name__.lower()
        if status_code == 429 or "ratelimit" in name:
            kind, retryable = ProviderErrorKind.RATE_LIMIT, True
        elif status_code in {408, 500, 502, 503, 504} or "timeout" in name or "connection" in name:
            kind, retryable = ProviderErrorKind.TRANSIENT, True
        elif status_code in {401, 403} or "authentication" in name or "permission" in name:
            kind, retryable = ProviderErrorKind.AUTH, False
        elif status_code in {400, 404, 422} or "badrequest" in name:
            kind, retryable = ProviderErrorKind.INVALID_REQUEST, False
        else:
            kind, retryable = ProviderErrorKind.UNKNOWN, False

        return ProviderError(
            provider=self.name,
            message=str(exc),
            code=f"OPENAI_{kind.value.upper()}",
            kind=kind,
            retryable=retryable,
            status_code=status_code,
        )

    async def chat(self, request: ProviderChatInput) -> ProviderChatResult:
        client = self._get_client()
        kwargs: dict[str, object] = {
            "model": request.model,
            "input": request.messages,
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens

        try:
            response = await client.responses.create(**kwargs)
            return ProviderChatResult(
                content=response.output_text,
                usage=self._usage(response),
                provider_request_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc

    async def structured(self, request: ProviderStructuredInput) -> ProviderChatResult:
        client = self._get_client()
        kwargs: dict[str, object] = {
            "model": request.model,
            "input": request.messages,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name,
                    "schema": request.response_schema,
                    "strict": True,
                }
            },
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_output_tokens"] = request.max_output_tokens

        try:
            response = await client.responses.create(**kwargs)
            content = response.output_text
            return ProviderChatResult(
                content=content,
                structured_data=json.loads(content),
                usage=self._usage(response),
                provider_request_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc

    async def embed(self, request: ProviderEmbeddingInput) -> ProviderEmbeddingResult:
        client = self._get_client()
        try:
            response = await client.embeddings.create(model=request.model, input=request.inputs)
            embeddings = [list(item.embedding) for item in response.data]
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            total_tokens = int(getattr(usage, "total_tokens", 0) or input_tokens) if usage else 0
            return ProviderEmbeddingResult(
                embeddings=embeddings,
                usage=ProviderUsage(input_tokens=input_tokens, total_tokens=total_tokens),
                provider_request_id=getattr(response, "id", None),
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise self._classify(exc) from exc
```

The OpenAI adapter uses `AsyncOpenAI`, so it can await SDK operations directly. Notice `max_retries=0`: this is intentional.

```python
self._client = AsyncOpenAI(
    api_key=self.api_key,
    timeout=self.timeout_seconds,
    max_retries=0,
)
```

Why disable SDK retries? Because Project 3 is specifically demonstrating a gateway-owned retry policy. If both the SDK and gateway retry independently, one logical request can make more provider calls than the gateway attempt log suggests.

The structured-output method translates the same normalized JSON Schema into OpenAI Responses API `text.format` configuration. This is the adapter pattern in action: same gateway operation, different SDK arguments.

## 9. Provider registry

**File: `app/providers/registry.py`**

```python
from functools import lru_cache

from app.core.config import get_settings
from app.providers.base import LLMProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider


class ProviderRegistry:
    def __init__(self, providers: list[LLMProvider]) -> None:
        self._providers = {provider.name: provider for provider in providers}

    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)

    def all(self) -> list[LLMProvider]:
        return list(self._providers.values())

    def configured(self) -> list[LLMProvider]:
        return [provider for provider in self._providers.values() if provider.configured]


@lru_cache
def get_provider_registry() -> ProviderRegistry:
    settings = get_settings()
    return ProviderRegistry([
        GeminiProvider(settings.gemini_api_key),
        OpenAIProvider(settings.openai_api_key, settings.gateway_timeout_seconds),
    ])
```

The registry is a lookup table from provider name to adapter object.

```python
self._providers = {provider.name: provider for provider in providers}
```

Without a registry, routing code often grows into `if provider == ... elif provider == ...`. The registry lets routing ask for a provider by name without importing every concrete adapter.

`@lru_cache` makes the default registry process-scoped. The same provider objects and SDK clients can be reused between requests.

## 10. RoutingService: separating policy from execution

**File: `app/services/routing_service.py`**

```python
from dataclasses import dataclass

from fastapi import status

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.providers.base import ProviderCapability
from app.providers.registry import ProviderRegistry, get_provider_registry


@dataclass(frozen=True)
class RouteTarget:
    provider: str
    model: str


class RoutingService:
    def __init__(self, settings: Settings, registry: ProviderRegistry) -> None:
        self.settings = settings
        self.registry = registry
        self._aliases = {
            "fast": settings.model_alias_fast,
            "reasoning": settings.model_alias_reasoning,
            "cheap": settings.model_alias_cheap,
            "embedding": settings.model_alias_embedding,
        }

    @staticmethod
    def parse_chain(chain: str) -> list[RouteTarget]:
        targets: list[RouteTarget] = []
        for raw_target in chain.split(","):
            value = raw_target.strip()
            if not value:
                continue
            if ":" not in value:
                raise AppError(
                    message=f"Invalid route target: {value}",
                    code="MODEL_ROUTE_INVALID",
                    status_code=500,
                )
            provider, model = value.split(":", 1)
            targets.append(RouteTarget(provider=provider.strip(), model=model.strip()))
        return targets

    def all_aliases(self) -> dict[str, list[RouteTarget]]:
        return {alias: self.parse_chain(chain) for alias, chain in self._aliases.items()}

    def resolve(
        self,
        *,
        alias: str,
        capability: ProviderCapability,
        provider_hint: str | None = None,
    ) -> list[RouteTarget]:
        chain = self._aliases.get(alias)
        if chain is None:
            raise AppError(
                message=f"Unknown model alias: {alias}",
                code="MODEL_ALIAS_UNKNOWN",
                status_code=status.HTTP_400_BAD_REQUEST,
                details={"available_aliases": sorted(self._aliases)},
            )

        resolved: list[RouteTarget] = []
        for target in self.parse_chain(chain):
            if provider_hint and target.provider != provider_hint:
                continue
            provider = self.registry.get(target.provider)
            if provider is None or not provider.configured or not provider.supports(capability):
                continue
            resolved.append(target)

        if not resolved:
            raise AppError(
                message="No configured provider can serve this model alias and capability.",
                code="NO_PROVIDER_AVAILABLE",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                details={
                    "alias": alias,
                    "capability": capability.value,
                    "provider_hint": provider_hint,
                },
            )
        return resolved


def get_routing_service() -> RoutingService:
    return RoutingService(get_settings(), get_provider_registry())
```

`RoutingService` does not call an LLM. It only decides which routes are eligible and in what order.

```text
fast -> [
    RouteTarget("gemini", "gemini-3.5-flash"),
    RouteTarget("openai", "gpt-5.6-terra"),
]
```

`parse_chain()` converts configuration text into typed `RouteTarget` objects. `resolve()` applies three filters: provider hint, provider configuration, and provider capability.

```python
if provider_hint and target.provider != provider_hint:
    continue

provider = self.registry.get(target.provider)
if provider is None or not provider.configured or not provider.supports(capability):
    continue
```

This is an example of policy being deterministic and testable without making network requests.

## 11. The GatewayService execution loop

**File: `app/services/gateway_service.py`**

```python
import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import TypeVar

from fastapi import status

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.pricing import PricingCatalog
from app.providers.base import (
    ProviderCapability,
    ProviderChatInput,
    ProviderChatResult,
    ProviderEmbeddingInput,
    ProviderEmbeddingResult,
    ProviderStructuredInput,
    ProviderUsage,
)
from app.providers.exceptions import ProviderError, ProviderErrorKind
from app.providers.registry import ProviderRegistry
from app.schemas.gateway import (
    AttemptData,
    ChatData,
    ChatRequest,
    EmbeddingData,
    EmbeddingRequest,
    StructuredChatData,
    StructuredChatRequest,
    TokenUsageData,
)
from app.services.routing_service import RouteTarget, RoutingService
from app.services.usage_service import UsageService

ResultT = TypeVar("ResultT", ProviderChatResult, ProviderEmbeddingResult)


@dataclass(frozen=True)
class GatewayExecution:
    target: RouteTarget
    result: ProviderChatResult | ProviderEmbeddingResult
    attempts: list[AttemptData]
    fallback_used: bool
    estimated_cost_usd: float | None


class GatewayService:
    def __init__(
        self,
        *,
        settings: Settings,
        registry: ProviderRegistry,
        routing: RoutingService,
        usage_service: UsageService,
        pricing: PricingCatalog,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.routing = routing
        self.usage_service = usage_service
        self.pricing = pricing
        self.sleep = sleep

    def _validate_messages(self, messages: list[dict[str, str]]) -> None:
        if len(messages) > self.settings.max_messages:
            raise AppError(
                message="Too many messages in one gateway request.",
                code="MESSAGE_LIMIT_EXCEEDED",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={"max_messages": self.settings.max_messages},
            )
        too_long = [
            index
            for index, item in enumerate(messages)
            if len(item["content"]) > self.settings.max_message_chars
        ]
        if too_long:
            raise AppError(
                message="One or more messages exceed the configured character limit.",
                code="MESSAGE_TOO_LARGE",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={
                    "max_message_chars": self.settings.max_message_chars,
                    "message_indexes": too_long,
                },
            )

    def _cost(self, target: RouteTarget, usage: ProviderUsage) -> float | None:
        return self.pricing.estimate_cost(
            provider=target.provider,
            model=target.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cached_input_tokens,
        )

    async def _execute(
        self,
        *,
        request_id: str,
        operation: str,
        model_alias: str,
        targets: list[RouteTarget],
        operation_factory: Callable[[RouteTarget], Awaitable[ResultT]],
    ) -> GatewayExecution:
        attempt_rows: list[AttemptData] = []
        logical_attempt_number = 0
        last_error: ProviderError | None = None

        for target_index, target in enumerate(targets):
            provider = self.registry.get(target.provider)
            if provider is None:
                continue

            for retry_index in range(self.settings.gateway_max_retries + 1):
                logical_attempt_number += 1
                started = perf_counter()
                try:
                    result = await asyncio.wait_for(
                        operation_factory(target),
                        timeout=self.settings.gateway_timeout_seconds,
                    )
                    latency_ms = round((perf_counter() - started) * 1000, 2)
                    cost = self._cost(target, result.usage)
                    row = AttemptData(
                        provider=target.provider,
                        model=target.model,
                        attempt_number=logical_attempt_number,
                        status="success",
                        latency_ms=latency_ms,
                    )
                    attempt_rows.append(row)
                    self.usage_service.record(
                        request_id=request_id,
                        operation=operation,
                        model_alias=model_alias,
                        provider=target.provider,
                        model=target.model,
                        attempt_number=logical_attempt_number,
                        success=True,
                        error_code=None,
                        latency_ms=latency_ms,
                        usage=result.usage,
                        estimated_cost_usd=cost,
                    )
                    return GatewayExecution(
                        target=target,
                        result=result,
                        attempts=attempt_rows,
                        fallback_used=target_index > 0,
                        estimated_cost_usd=cost,
                    )
                except TimeoutError:
                    error = ProviderError(
                        provider=target.provider,
                        message="Provider call exceeded the gateway timeout.",
                        code="PROVIDER_TIMEOUT",
                        kind=ProviderErrorKind.TRANSIENT,
                        retryable=True,
                        status_code=504,
                    )
                except ProviderError as exc:
                    error = exc

                latency_ms = round((perf_counter() - started) * 1000, 2)
                last_error = error
                attempt_rows.append(
                    AttemptData(
                        provider=target.provider,
                        model=target.model,
                        attempt_number=logical_attempt_number,
                        status="failed",
                        latency_ms=latency_ms,
                        error_code=error.code,
                    )
                )
                self.usage_service.record(
                    request_id=request_id,
                    operation=operation,
                    model_alias=model_alias,
                    provider=target.provider,
                    model=target.model,
                    attempt_number=logical_attempt_number,
                    success=False,
                    error_code=error.code,
                    latency_ms=latency_ms,
                    usage=ProviderUsage(),
                    estimated_cost_usd=None,
                )

                if error.kind == ProviderErrorKind.INVALID_REQUEST:
                    raise AppError(
                        message="The provider rejected the request as invalid.",
                        code="PROVIDER_INVALID_REQUEST",
                        status_code=status.HTTP_400_BAD_REQUEST,
                        details={"provider": target.provider, "provider_code": error.code},
                    ) from error

                can_retry = error.retryable and retry_index < self.settings.gateway_max_retries
                if can_retry:
                    delay = min(
                        self.settings.gateway_backoff_base_seconds * (2**retry_index),
                        self.settings.gateway_backoff_max_seconds,
                    )
                    jitter = random.uniform(0, delay * 0.25) if delay > 0 else 0.0
                    await self.sleep(delay + jitter)
                    continue
                break

        raise AppError(
            message="All configured provider routes failed.",
            code="ALL_PROVIDERS_FAILED",
            status_code=status.HTTP_502_BAD_GATEWAY,
            details={
                "attempts": [row.model_dump() for row in attempt_rows],
                "last_provider_code": last_error.code if last_error else None,
            },
        ) from last_error

    @staticmethod
    def _usage_data(usage: ProviderUsage, cost: float | None) -> TokenUsageData:
        return TokenUsageData(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_input_tokens=usage.cached_input_tokens,
            estimated_cost_usd=cost,
        )

    async def chat(self, request: ChatRequest, request_id: str) -> ChatData:
        messages = [item.model_dump() for item in request.messages]
        self._validate_messages(messages)
        targets = self.routing.resolve(
            alias=request.model,
            capability=ProviderCapability.CHAT,
            provider_hint=request.provider,
        )

        async def operation(target: RouteTarget) -> ProviderChatResult:
            provider = self.registry.get(target.provider)
            assert provider is not None
            return await provider.chat(
                ProviderChatInput(
                    model=target.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                )
            )

        execution = await self._execute(
            request_id=request_id,
            operation="chat",
            model_alias=request.model,
            targets=targets,
            operation_factory=operation,
        )
        result = execution.result
        assert isinstance(result, ProviderChatResult)
        return ChatData(
            request_id=request_id,
            model_alias=request.model,
            provider=execution.target.provider,
            model=execution.target.model,
            content=result.content,
            fallback_used=execution.fallback_used,
            usage=self._usage_data(result.usage, execution.estimated_cost_usd),
            attempts=execution.attempts,
        )

    async def structured(
        self, request: StructuredChatRequest, request_id: str
    ) -> StructuredChatData:
        messages = [item.model_dump() for item in request.messages]
        self._validate_messages(messages)
        targets = self.routing.resolve(
            alias=request.model,
            capability=ProviderCapability.STRUCTURED,
            provider_hint=request.provider,
        )

        async def operation(target: RouteTarget) -> ProviderChatResult:
            provider = self.registry.get(target.provider)
            assert provider is not None
            return await provider.structured(
                ProviderStructuredInput(
                    model=target.model,
                    messages=messages,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    schema_name=request.schema_name,
                    response_schema=request.response_schema,
                )
            )

        execution = await self._execute(
            request_id=request_id,
            operation="structured",
            model_alias=request.model,
            targets=targets,
            operation_factory=operation,
        )
        result = execution.result
        assert isinstance(result, ProviderChatResult)
        if result.structured_data is None:
            raise AppError(
                message="Provider returned no structured data.",
                code="STRUCTURED_OUTPUT_MISSING",
                status_code=status.HTTP_502_BAD_GATEWAY,
            )
        return StructuredChatData(
            request_id=request_id,
            model_alias=request.model,
            provider=execution.target.provider,
            model=execution.target.model,
            data=result.structured_data,
            fallback_used=execution.fallback_used,
            usage=self._usage_data(result.usage, execution.estimated_cost_usd),
            attempts=execution.attempts,
        )

    async def embed(self, request: EmbeddingRequest, request_id: str) -> EmbeddingData:
        if len(request.inputs) > self.settings.max_embedding_inputs:
            raise AppError(
                message="Too many embedding inputs.",
                code="EMBEDDING_INPUT_LIMIT_EXCEEDED",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={"max_embedding_inputs": self.settings.max_embedding_inputs},
            )
        if any(len(item) > self.settings.max_embedding_chars for item in request.inputs):
            raise AppError(
                message="One or more embedding inputs exceed the configured character limit.",
                code="EMBEDDING_INPUT_TOO_LARGE",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                details={"max_embedding_chars": self.settings.max_embedding_chars},
            )

        targets = self.routing.resolve(
            alias=request.model,
            capability=ProviderCapability.EMBEDDINGS,
            provider_hint=request.provider,
        )

        async def operation(target: RouteTarget) -> ProviderEmbeddingResult:
            provider = self.registry.get(target.provider)
            assert provider is not None
            return await provider.embed(
                ProviderEmbeddingInput(model=target.model, inputs=request.inputs)
            )

        execution = await self._execute(
            request_id=request_id,
            operation="embeddings",
            model_alias=request.model,
            targets=targets,
            operation_factory=operation,
        )
        result = execution.result
        assert isinstance(result, ProviderEmbeddingResult)
        dimensions = len(result.embeddings[0]) if result.embeddings else 0
        return EmbeddingData(
            request_id=request_id,
            model_alias=request.model,
            provider=execution.target.provider,
            model=execution.target.model,
            dimensions=dimensions,
            embeddings=result.embeddings,
            fallback_used=execution.fallback_used,
            usage=self._usage_data(result.usage, execution.estimated_cost_usd),
            attempts=execution.attempts,
        )
```

This is the most important file in Project 3. Read `_execute()` slowly. It has two nested loops.

```python
for target_index, target in enumerate(targets):
    ...
    for retry_index in range(self.settings.gateway_max_retries + 1):
        ...
```

Outer loop = fallback routes. Inner loop = retries on the same route.

```text
Gemini attempt 1 fails transiently
Gemini attempt 2 fails transiently
Gemini retries exhausted
OpenAI attempt 3 succeeds
```

`logical_attempt_number` counts actual gateway attempts across both loops. This gives one ordered audit trail.

## 12. Gateway timeout

```python
result = await asyncio.wait_for(
    operation_factory(target),
    timeout=self.settings.gateway_timeout_seconds,
)
```

`asyncio.wait_for` places a deadline around the provider operation. The gateway therefore has a timeout boundary even if provider SDK behavior differs.

A timeout is normalized into a retryable transient `ProviderError`. This lets the same retry/fallback policy handle it.

## 13. Exponential backoff and jitter

```python
delay = min(
    base_seconds * (2**retry_index),
    max_seconds,
)
jitter = random.uniform(0, delay * 0.25)
await self.sleep(delay + jitter)
```

Exponential means the delay grows multiplicatively. With base `0.25`, the raw sequence is 0.25, 0.50, 1.00, 2.00, and so on until the configured cap.

Jitter adds randomness. Imagine 1,000 gateway replicas receive a provider 429 at the same moment. Without jitter they may all sleep exactly one second and hit the provider again at exactly the same moment. Jitter spreads those retries over a small window.

## 14. Fail fast vs fallback

```python
if error.kind == ProviderErrorKind.INVALID_REQUEST:
    raise AppError(...)
```

An invalid request should fail fast. Fallback is not automatically correct for every error. If the caller sent malformed input, switching vendors hides the real bug and creates unnecessary cost.

Transient failures, rate limits, authentication/configuration failures, and unknown provider-specific failures may eventually move to the next target after same-provider retries are exhausted. The exact policy is centralized here instead of scattered across adapters.

## 15. Operation factory: one execution engine for three operations

```python
async def operation(target: RouteTarget) -> ProviderChatResult:
    provider = self.registry.get(target.provider)
    return await provider.chat(...)


execution = await self._execute(
    ...,
    operation_factory=operation,
)
```

`_execute()` should not know the details of chat, structured output, or embeddings. Each public method creates a small operation function and passes it into the execution engine.

This is higher-order programming: a function receives another callable as data. The benefit is one retry/fallback implementation instead of three duplicated implementations.

## 16. Cost accounting is separate from token normalization

**File: `app/core/pricing.py`**

```python
import json
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import AppError


@dataclass(frozen=True)
class PriceEntry:
    input_per_million_usd: float | None
    output_per_million_usd: float | None
    cached_input_per_million_usd: float | None = None


class PricingCatalog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._prices = self._load(path)

    @staticmethod
    def _load(path: Path) -> dict[tuple[str, str], PriceEntry]:
        if not path.exists():
            raise AppError(
                message=f"Pricing catalog not found: {path}",
                code="PRICING_CATALOG_MISSING",
                status_code=500,
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        prices: dict[tuple[str, str], PriceEntry] = {}

        for provider, models in payload["providers"].items():
            for model, values in models.items():
                prices[(provider, model)] = PriceEntry(
                    input_per_million_usd=values.get("input_per_million_usd"),
                    output_per_million_usd=values.get("output_per_million_usd"),
                    cached_input_per_million_usd=values.get("cached_input_per_million_usd"),
                )
        return prices

    def get(self, provider: str, model: str) -> PriceEntry | None:
        return self._prices.get((provider, model))

    def estimate_cost(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> float | None:
        entry = self.get(provider, model)
        if entry is None or entry.input_per_million_usd is None:
            return None

        uncached_input = max(input_tokens - cached_input_tokens, 0)
        input_cost = uncached_input / 1_000_000 * entry.input_per_million_usd

        cached_rate = entry.cached_input_per_million_usd
        if cached_rate is None:
            cached_rate = entry.input_per_million_usd
        cached_cost = cached_input_tokens / 1_000_000 * cached_rate

        output_cost = 0.0
        if entry.output_per_million_usd is not None:
            output_cost = output_tokens / 1_000_000 * entry.output_per_million_usd

        return round(input_cost + cached_cost + output_cost, 10)
```

Provider adapters report tokens. `PricingCatalog` maps provider/model names to configured prices. `GatewayService` combines those pieces.

```text
provider result
  -> normalized token usage
  -> PricingCatalog.get(provider, model)
  -> estimated cost
```

The formula operates on prices per one million tokens.

```python
input_cost = input_tokens / 1_000_000 * input_rate
output_cost = output_tokens / 1_000_000 * output_rate
```

Cached input tokens can have a separate price. The code subtracts cached input from total input before calculating the normal input component, then prices cached tokens separately.

The catalog returns `None` when a model price is unknown. This is deliberate. A cost observability system should not silently invent a price.

## 17. The pricing JSON file

**File: `config/pricing.json`**

```json
{
  "catalog_version": "2026-07-10",
  "currency": "USD",
  "unit": "per_1m_tokens",
  "providers": {
    "gemini": {
      "gemini-3.5-flash": {
        "input_per_million_usd": 1.5,
        "output_per_million_usd": 9.0,
        "cached_input_per_million_usd": 0.15
      },
      "gemini-3.1-flash-lite": {
        "input_per_million_usd": 0.25,
        "output_per_million_usd": 1.5,
        "cached_input_per_million_usd": 0.025
      },
      "gemini-embedding-001": {
        "input_per_million_usd": 0.15,
        "output_per_million_usd": 0.0
      }
    },
    "openai": {
      "gpt-5.6-sol": {
        "input_per_million_usd": 5.0,
        "output_per_million_usd": 30.0,
        "cached_input_per_million_usd": 0.5
      },
      "gpt-5.6-terra": {
        "input_per_million_usd": 0.75,
        "output_per_million_usd": 4.5,
        "cached_input_per_million_usd": 0.075
      },
      "gpt-5.6-luna": {
        "input_per_million_usd": 0.2,
        "output_per_million_usd": 1.25,
        "cached_input_per_million_usd": 0.02
      },
      "text-embedding-3-small": {
        "input_per_million_usd": null,
        "output_per_million_usd": null
      }
    }
  }
}
```

This is a dated pricing snapshot. It is configuration, not source code and not a billing invoice. Provider prices can change. The public README tells users to update this file.

Why keep a date? Because a number such as `1.5` is meaningless operationally if nobody knows when the catalog was last reviewed.

## 18. Request context middleware

**File: `app/core/middleware.py`**

```python
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        started = perf_counter()

        response = await call_next(request)

        latency_ms = round((perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        logger.info(
            "%s %s -> %s | %.2fms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response
```

Middleware wraps every HTTP request before and after route execution.

```python
request_id = request.headers.get("X-Request-ID") or str(uuid4())
```

This gives two modes. A caller with an existing trace ID can send it. A caller without one gets a generated UUID.

```python
request.state.request_id = request_id
```

`request.state` is per-request storage. Endpoints and exception handlers can read the same request ID without adding a Pydantic request field.

The middleware also adds `X-Response-Time-Ms`. This is end-to-end API latency, not provider-only latency. Provider attempt latency is measured separately inside `GatewayService`.

## 19. Why both API latency and provider latency matter

```text
X-Response-Time-Ms = middleware end-to-end time
AttemptData.latency_ms = one provider attempt time
```

If API latency is 2,000 ms but provider latency is 500 ms, 1,500 ms is happening elsewhere: database work, serialization, application queues, or another layer. Two timing scopes help localize latency.

## 20. SQLAlchemy usage event model

**File: `app/db/models.py`**

```python
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    operation: Mapped[str] = mapped_column(String(32), index=True)
    model_alias: Mapped[str] = mapped_column(String(64), index=True)
    provider: Mapped[str] = mapped_column(String(32), index=True)
    model: Mapped[str] = mapped_column(String(128), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    success: Mapped[bool] = mapped_column(Boolean, index=True)
    error_code: Mapped[str | None] = mapped_column(String(96), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        index=True,
    )
```

Project 1 created a database engine but did not persist a real application entity. Project 3 adds `UsageEvent`, a real SQLAlchemy mapped model.

One row represents one provider attempt, not one HTTP request. That data model is intentional because fallback can produce multiple provider attempts for one request ID.

```text
request_id = "abc"

row 1: Gemini attempt 1 failed
row 2: Gemini attempt 2 failed
row 3: OpenAI attempt 3 succeeded
```

`request_id` groups the three rows into one logical request.

## 21. Repository pattern

**File: `app/repositories/usage_repository.py`**

```python
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import UsageEvent


class UsageRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, event: UsageEvent) -> UsageEvent:
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def recent(self, *, since: datetime, limit: int) -> list[UsageEvent]:
        statement = (
            select(UsageEvent)
            .where(UsageEvent.created_at >= since)
            .order_by(UsageEvent.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement))

    def totals(self, *, since: datetime) -> dict[str, int | float]:
        statement = select(
            func.count(UsageEvent.id),
            func.sum(UsageEvent.input_tokens),
            func.sum(UsageEvent.output_tokens),
            func.sum(UsageEvent.total_tokens),
            func.sum(UsageEvent.estimated_cost_usd),
            func.sum(func.cast(UsageEvent.success, UsageEvent.id.type)),
        ).where(UsageEvent.created_at >= since)
        row = self.db.execute(statement).one()
        return {
            "attempts": int(row[0] or 0),
            "input_tokens": int(row[1] or 0),
            "output_tokens": int(row[2] or 0),
            "total_tokens": int(row[3] or 0),
            "estimated_cost_usd": float(row[4] or 0.0),
            "successful_attempts": int(row[5] or 0),
        }

    def breakdown(self, *, since: datetime) -> list[dict[str, object]]:
        statement = (
            select(
                UsageEvent.provider,
                UsageEvent.model,
                func.count(UsageEvent.id),
                func.sum(UsageEvent.total_tokens),
                func.sum(UsageEvent.estimated_cost_usd),
            )
            .where(UsageEvent.created_at >= since)
            .group_by(UsageEvent.provider, UsageEvent.model)
            .order_by(UsageEvent.provider, UsageEvent.model)
        )
        rows = self.db.execute(statement).all()
        return [
            {
                "provider": row[0],
                "model": row[1],
                "attempts": int(row[2] or 0),
                "total_tokens": int(row[3] or 0),
                "estimated_cost_usd": float(row[4] or 0.0),
            }
            for row in rows
        ]
```

`UsageRepository` owns SQL queries. `UsageService` should not know SQLAlchemy `select`, `func.sum`, or `group_by` syntax.

```text
UsageService
    asks: give me totals and recent events

UsageRepository
    knows: SELECT, SUM, COUNT, GROUP BY
```

This is a repository boundary. It becomes useful when query logic is nontrivial or when business services should remain database-technology-light.

## 22. Aggregation queries

```python
select(
    func.count(UsageEvent.id),
    func.sum(UsageEvent.input_tokens),
    func.sum(UsageEvent.output_tokens),
    func.sum(UsageEvent.total_tokens),
    func.sum(UsageEvent.estimated_cost_usd),
)
```

The database performs aggregation instead of loading every usage row into Python and summing manually. Databases are designed for this kind of grouped aggregate operation.

## 23. UsageService

**File: `app/services/usage_service.py`**

```python
from datetime import UTC, datetime, timedelta

from app.db.models import UsageEvent
from app.providers.base import ProviderUsage
from app.repositories.usage_repository import UsageRepository
from app.schemas.usage import (
    UsageBreakdownData,
    UsageData,
    UsageEventData,
    UsageTotalsData,
)


class UsageService:
    def __init__(self, repository: UsageRepository) -> None:
        self.repository = repository

    def record(
        self,
        *,
        request_id: str,
        operation: str,
        model_alias: str,
        provider: str,
        model: str,
        attempt_number: int,
        success: bool,
        error_code: str | None,
        latency_ms: float,
        usage: ProviderUsage,
        estimated_cost_usd: float | None,
    ) -> None:
        self.repository.add(
            UsageEvent(
                request_id=request_id,
                operation=operation,
                model_alias=model_alias,
                provider=provider,
                model=model,
                attempt_number=attempt_number,
                success=success,
                error_code=error_code,
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
        )

    def summary(self, *, days: int, limit: int) -> UsageData:
        since = datetime.now(UTC) - timedelta(days=days)
        totals = self.repository.totals(since=since)
        recent = self.repository.recent(since=since, limit=limit)
        breakdown = self.repository.breakdown(since=since)
        successful = int(totals["successful_attempts"])
        attempts = int(totals["attempts"])

        return UsageData(
            window_days=days,
            totals=UsageTotalsData(
                attempts=attempts,
                successful_attempts=successful,
                failed_attempts=attempts - successful,
                input_tokens=int(totals["input_tokens"]),
                output_tokens=int(totals["output_tokens"]),
                total_tokens=int(totals["total_tokens"]),
                estimated_cost_usd=round(float(totals["estimated_cost_usd"]), 10),
            ),
            breakdown=[UsageBreakdownData(**item) for item in breakdown],
            recent_events=[
                UsageEventData(
                    request_id=item.request_id,
                    operation=item.operation,
                    model_alias=item.model_alias,
                    provider=item.provider,
                    model=item.model,
                    attempt_number=item.attempt_number,
                    success=item.success,
                    error_code=item.error_code,
                    latency_ms=item.latency_ms,
                    total_tokens=item.total_tokens,
                    estimated_cost_usd=item.estimated_cost_usd,
                    created_at=item.created_at,
                )
                for item in recent
            ],
        )
```

`UsageService.record()` converts normalized gateway facts into a persistence model. `summary()` combines three repository queries into the public API schema.

Notice `failed_attempts` is derived.

```python
failed_attempts = attempts - successful_attempts
```

Derived values should not always become database columns. If a value can be reliably calculated from stored data, storing a second copy can create consistency problems.

## 24. Gateway request and response schemas

**File: `app/schemas/gateway.py`**

```python
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default="fast", min_length=1, max_length=64)
    messages: list[Message] = Field(min_length=1)
    provider: str | None = Field(default=None, max_length=32)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=100_000)


class StructuredChatRequest(ChatRequest):
    schema_name: str = Field(default="response", pattern=r"^[A-Za-z0-9_-]+$")
    response_schema: dict[str, Any]

    @field_validator("response_schema")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("response_schema must have top-level type='object'")
        if "properties" not in value:
            raise ValueError("response_schema must define properties")
        return value


class EmbeddingRequest(BaseModel):
    model: str = Field(default="embedding", min_length=1, max_length=64)
    inputs: list[str] = Field(min_length=1)
    provider: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_inputs(self) -> "EmbeddingRequest":
        if any(not item.strip() for item in self.inputs):
            raise ValueError("embedding inputs cannot be blank")
        return self


class TokenUsageData(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float | None


class AttemptData(BaseModel):
    provider: str
    model: str
    attempt_number: int
    status: Literal["success", "failed"]
    latency_ms: float
    error_code: str | None = None


class ChatData(BaseModel):
    request_id: str
    model_alias: str
    provider: str
    model: str
    content: str
    fallback_used: bool
    usage: TokenUsageData
    attempts: list[AttemptData]


class StructuredChatData(BaseModel):
    request_id: str
    model_alias: str
    provider: str
    model: str
    data: dict[str, Any]
    fallback_used: bool
    usage: TokenUsageData
    attempts: list[AttemptData]


class EmbeddingData(BaseModel):
    request_id: str
    model_alias: str
    provider: str
    model: str
    dimensions: int
    embeddings: list[list[float]]
    fallback_used: bool
    usage: TokenUsageData
    attempts: list[AttemptData]
```

The public API accepts aliases, not provider model IDs. `ChatRequest.model` defaults to `fast`. The optional `provider` is a routing hint for debugging/evaluation.

`StructuredChatRequest` validates only top-level object schemas in this starter project. That is a deliberate scope boundary. The gateway could later support arrays or more schema variants, but starting with object schemas keeps provider behavior easier to compare.

```python
if value.get("type") != "object":
    raise ValueError(...)
if "properties" not in value:
    raise ValueError(...)
```

Pydantic rejects the request before a provider call, so invalid schema shape costs zero provider tokens.

## 25. Attempt history in successful responses

```json
"attempts": [
  {"provider": "gemini", "status": "failed", "error_code": "..."},
  {"provider": "openai", "status": "success", "error_code": null}
]
```

Most application APIs hide retry behavior. This educational gateway exposes attempt history so you can observe the routing engine directly in Swagger.

For a public production API, you might hide raw provider names or error codes from end users and keep them only in logs/telemetry. Here they are intentionally visible for learning.

## 26. Health vs readiness

**File: `app/api/v1/endpoints/health.py`**

```python
from fastapi import APIRouter, Depends, Request, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.exceptions import AppError
from app.db.database import get_db
from app.providers.registry import ProviderRegistry, get_provider_registry
from app.schemas.common import SuccessResponse
from app.schemas.health import HealthData, ReadyData

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=SuccessResponse[HealthData])
def health(request: Request) -> SuccessResponse[HealthData]:
    return SuccessResponse(
        request_id=request.state.request_id,
        data=HealthData(app=settings.app_name, environment=settings.app_env),
    )


@router.get("/ready", response_model=SuccessResponse[ReadyData])
def ready(
    request: Request,
    db: Session = Depends(get_db),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> SuccessResponse[ReadyData]:
    db.execute(text("SELECT 1"))
    configured_count = len(registry.configured())
    if configured_count == 0:
        raise AppError(
            message="No LLM provider is configured.",
            code="NO_PROVIDER_CONFIGURED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return SuccessResponse(
        request_id=request.state.request_id,
        data=ReadyData(configured_providers=configured_count),
    )
```

`/health` answers: is the FastAPI process alive? It does not require provider credentials.

`/ready` answers: can this gateway reasonably accept LLM traffic? It checks the database and requires at least one configured provider.

```text
health = process liveness
ready  = dependency/configuration readiness
```

Keeping these separate matters in container orchestration. A process can be alive but not ready to serve useful traffic.

## 27. Provider metadata endpoint

**File: `app/api/v1/endpoints/metadata.py`**

```python
from fastapi import APIRouter, Depends, Request

from app.providers.registry import ProviderRegistry, get_provider_registry
from app.schemas.common import SuccessResponse
from app.schemas.provider import (
    ModelAliasData,
    ModelsData,
    ModelTargetData,
    ProviderData,
    ProvidersData,
)
from app.services.routing_service import RoutingService, get_routing_service

router = APIRouter()


@router.get("/providers", response_model=SuccessResponse[ProvidersData])
def providers(
    request: Request,
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> SuccessResponse[ProvidersData]:
    rows = [
        ProviderData(
            name=provider.name,
            configured=provider.configured,
            capabilities=sorted(capability.value for capability in provider.capabilities),
        )
        for provider in registry.all()
    ]
    return SuccessResponse(request_id=request.state.request_id, data=ProvidersData(providers=rows))


@router.get("/models", response_model=SuccessResponse[ModelsData])
def models(
    request: Request,
    routing: RoutingService = Depends(get_routing_service),
    registry: ProviderRegistry = Depends(get_provider_registry),
) -> SuccessResponse[ModelsData]:
    aliases = []
    for alias, targets in routing.all_aliases().items():
        aliases.append(
            ModelAliasData(
                alias=alias,
                targets=[
                    ModelTargetData(
                        provider=target.provider,
                        model=target.model,
                        configured=bool(
                            registry.get(target.provider)
                            and registry.get(target.provider).configured
                        ),
                    )
                    for target in targets
                ],
            )
        )
    return SuccessResponse(request_id=request.state.request_id, data=ModelsData(aliases=aliases))
```

`/providers` reports adapter capability and configuration state. It never returns API keys.

`/models` reports alias chains and whether each target's provider is configured. This makes routing policy inspectable without opening `.env` on the server.

## 28. Gateway dependency construction

**File: `app/api/v1/endpoints/gateway.py`**

```python
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.pricing import PricingCatalog
from app.db.database import get_db
from app.providers.registry import get_provider_registry
from app.repositories.usage_repository import UsageRepository
from app.schemas.common import SuccessResponse
from app.schemas.gateway import (
    ChatData,
    ChatRequest,
    EmbeddingData,
    EmbeddingRequest,
    StructuredChatData,
    StructuredChatRequest,
)
from app.services.gateway_service import GatewayService
from app.services.routing_service import get_routing_service
from app.services.usage_service import UsageService

router = APIRouter()


def get_gateway_service(db: Session = Depends(get_db)) -> GatewayService:
    settings = get_settings()
    return GatewayService(
        settings=settings,
        registry=get_provider_registry(),
        routing=get_routing_service(),
        usage_service=UsageService(UsageRepository(db)),
        pricing=PricingCatalog(settings.pricing_catalog_path),
    )


@router.post("/chat", response_model=SuccessResponse[ChatData])
async def chat(
    payload: ChatRequest,
    request: Request,
    service: GatewayService = Depends(get_gateway_service),
) -> SuccessResponse[ChatData]:
    data = await service.chat(payload, request.state.request_id)
    return SuccessResponse(request_id=request.state.request_id, data=data)


@router.post("/chat/structured", response_model=SuccessResponse[StructuredChatData])
async def structured_chat(
    payload: StructuredChatRequest,
    request: Request,
    service: GatewayService = Depends(get_gateway_service),
) -> SuccessResponse[StructuredChatData]:
    data = await service.structured(payload, request.state.request_id)
    return SuccessResponse(request_id=request.state.request_id, data=data)


@router.post("/embeddings", response_model=SuccessResponse[EmbeddingData])
async def embeddings(
    payload: EmbeddingRequest,
    request: Request,
    service: GatewayService = Depends(get_gateway_service),
) -> SuccessResponse[EmbeddingData]:
    data = await service.embed(payload, request.state.request_id)
    return SuccessResponse(request_id=request.state.request_id, data=data)
```

`get_gateway_service()` is a FastAPI dependency factory. It assembles the gateway from five collaborating objects: settings, registry, routing service, usage service, and pricing catalog.

```python
GatewayService(
    settings=settings,
    registry=get_provider_registry(),
    routing=get_routing_service(),
    usage_service=UsageService(UsageRepository(db)),
    pricing=PricingCatalog(...),
)
```

This is manual dependency injection. The class does not construct its own database session or provider registry. Its collaborators are passed in. That is why tests can pass fake providers and a test database.

## 29. Usage endpoint

**File: `app/api/v1/endpoints/usage.py`**

```python
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.repositories.usage_repository import UsageRepository
from app.schemas.common import SuccessResponse
from app.schemas.usage import UsageData
from app.services.usage_service import UsageService

router = APIRouter()


def get_usage_service(db: Session = Depends(get_db)) -> UsageService:
    return UsageService(UsageRepository(db))


@router.get("/usage", response_model=SuccessResponse[UsageData])
def usage(
    request: Request,
    days: int = Query(default=7, ge=1, le=365),
    limit: int = Query(default=100, ge=1, le=1000),
    service: UsageService = Depends(get_usage_service),
) -> SuccessResponse[UsageData]:
    return SuccessResponse(
        request_id=request.state.request_id,
        data=service.summary(days=days, limit=limit),
    )
```

The endpoint uses query parameters instead of request JSON because it is a read operation.

```text
/usage?days=7&limit=100
```

`days` defines the analytics time window. `limit` only limits recent event rows; totals and breakdown still aggregate the entire selected window.

## 30. Main application lifespan

**File: `app/main.py`**

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware
from app.db.database import init_db

settings = get_settings()
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    init_db()
    logger.info("Database initialized")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        debug=settings.app_debug,
        lifespan=lifespan,
        description=(
            "Provider-neutral LLM gateway with alias routing, retries, fallback, "
            "normalized usage, cost estimation, structured output, and embeddings."
        ),
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
```

Project 3 uses FastAPI lifespan to initialize database tables before serving requests.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield
```

Code before `yield` is startup logic. Code after `yield` would be shutdown logic. There is no shutdown cleanup required for the synchronous SQLAlchemy engine in this small project.

## 31. SQLite test configuration and StaticPool

**File: `app/db/database.py`**

```python
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.base import Base

settings = get_settings()

engine_kwargs: dict[str, object] = {}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
if settings.database_url == "sqlite://":
    engine_kwargs["poolclass"] = StaticPool

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Tests set `DATABASE_URL=sqlite://`. That is an in-memory SQLite database.

An in-memory SQLite database normally has a lifetime tied to a connection. `StaticPool` makes SQLAlchemy reuse one connection so different sessions in the test process see the same in-memory database.

```python
if settings.database_url == "sqlite://":
    engine_kwargs["poolclass"] = StaticPool
```

This is why the test database works across repository and API sessions.

## 32. Test fakes are provider fakes, not SDK mocks everywhere

**File: `tests/fakes.py`**

```python
from collections import deque

from app.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderChatInput,
    ProviderChatResult,
    ProviderEmbeddingInput,
    ProviderEmbeddingResult,
    ProviderStructuredInput,
    ProviderUsage,
)


class FakeProvider(LLMProvider):
    capabilities = frozenset({
        ProviderCapability.CHAT,
        ProviderCapability.STRUCTURED,
        ProviderCapability.EMBEDDINGS,
    })

    def __init__(
        self, name: str, outcomes: list[object] | None = None, configured: bool = True
    ) -> None:
        self.name = name
        self._configured = configured
        self.outcomes = deque(outcomes or [])
        self.calls = 0

    @property
    def configured(self) -> bool:
        return self._configured

    def _next(self, default: object) -> object:
        self.calls += 1
        if self.outcomes:
            value = self.outcomes.popleft()
            if isinstance(value, Exception):
                raise value
            return value
        return default

    async def chat(self, request: ProviderChatInput) -> ProviderChatResult:
        del request
        return self._next(
            ProviderChatResult(
                content=f"reply from {self.name}",
                usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )

    async def structured(self, request: ProviderStructuredInput) -> ProviderChatResult:
        del request
        return self._next(
            ProviderChatResult(
                content='{"answer":"ok"}',
                structured_data={"answer": "ok"},
                usage=ProviderUsage(input_tokens=12, output_tokens=4, total_tokens=16),
            )
        )

    async def embed(self, request: ProviderEmbeddingInput) -> ProviderEmbeddingResult:
        return self._next(
            ProviderEmbeddingResult(
                embeddings=[[0.1, 0.2, 0.3] for _ in request.inputs],
                usage=ProviderUsage(input_tokens=8, total_tokens=8),
            )
        )
```

`FakeProvider` implements the same `LLMProvider` interface as Gemini and OpenAI. Gateway tests therefore exercise real routing and retry logic without knowing provider SDK details.

The `outcomes` deque lets a test script provider behavior.

```python
outcomes = [
    transient_error,
    transient_error,
    successful_result,
]
```

Each call removes the next outcome. This makes retry tests deterministic.

## 33. Retry-before-fallback test

**File: `tests/test_gateway_service.py`**

```python
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.pricing import PricingCatalog
from app.db.database import SessionLocal
from app.providers.base import ProviderChatResult, ProviderUsage
from app.providers.exceptions import ProviderError, ProviderErrorKind
from app.providers.registry import ProviderRegistry
from app.repositories.usage_repository import UsageRepository
from app.schemas.gateway import ChatRequest, Message
from app.services.gateway_service import GatewayService
from app.services.routing_service import RoutingService
from app.services.usage_service import UsageService
from tests.fakes import FakeProvider


def build_service(db: Session, providers: list[FakeProvider]) -> GatewayService:
    settings = Settings(
        MODEL_ALIAS_FAST="primary:model-a,fallback:model-b",
        MODEL_ALIAS_REASONING="primary:model-r",
        MODEL_ALIAS_CHEAP="primary:model-c",
        MODEL_ALIAS_EMBEDDING="fallback:embed-b",
        GATEWAY_MAX_RETRIES=1,
        GATEWAY_BACKOFF_BASE_SECONDS=0,
        GATEWAY_BACKOFF_MAX_SECONDS=0,
    )
    registry = ProviderRegistry(providers)
    return GatewayService(
        settings=settings,
        registry=registry,
        routing=RoutingService(settings, registry),
        usage_service=UsageService(UsageRepository(db)),
        pricing=PricingCatalog(Path("config/pricing.json")),
        sleep=lambda _: noop(),
    )


async def noop() -> None:
    return None


@pytest.mark.anyio
async def test_retry_succeeds_on_same_provider_before_fallback() -> None:
    transient = ProviderError(
        provider="primary",
        message="temporary",
        code="PRIMARY_TRANSIENT",
        kind=ProviderErrorKind.TRANSIENT,
        retryable=True,
    )
    primary = FakeProvider(
        "primary",
        outcomes=[
            transient,
            ProviderChatResult(
                content="recovered",
                usage=ProviderUsage(input_tokens=10, output_tokens=2, total_tokens=12),
            ),
        ],
    )
    fallback = FakeProvider("fallback")
    with SessionLocal() as db:
        service = build_service(db, [primary, fallback])
        result = await service.chat(
            ChatRequest(messages=[Message(role="user", content="hello")]),
            "req-retry",
        )
        assert result.content == "recovered"
        assert result.fallback_used is False
        assert len(result.attempts) == 2
        assert result.attempts[0].status == "failed"
        assert result.attempts[1].status == "success"
        assert fallback.calls == 0


@pytest.mark.anyio
async def test_fallback_provider_used_after_primary_retries_fail() -> None:
    transient = ProviderError(
        provider="primary",
        message="temporary",
        code="PRIMARY_TRANSIENT",
        kind=ProviderErrorKind.TRANSIENT,
        retryable=True,
    )
    primary = FakeProvider("primary", outcomes=[transient, transient])
    fallback = FakeProvider("fallback")
    with SessionLocal() as db:
        service = build_service(db, [primary, fallback])
        result = await service.chat(
            ChatRequest(messages=[Message(role="user", content="hello")]),
            "req-fallback",
        )
        assert result.provider == "fallback"
        assert result.fallback_used is True
        assert len(result.attempts) == 3


@pytest.mark.anyio
async def test_invalid_provider_request_does_not_retry_or_fallback() -> None:
    invalid = ProviderError(
        provider="primary",
        message="bad input",
        code="PRIMARY_INVALID",
        kind=ProviderErrorKind.INVALID_REQUEST,
        retryable=False,
    )
    primary = FakeProvider("primary", outcomes=[invalid])
    fallback = FakeProvider("fallback")
    with SessionLocal() as db:
        service = build_service(db, [primary, fallback])
        with pytest.raises(Exception) as exc_info:
            await service.chat(
                ChatRequest(messages=[Message(role="user", content="hello")]),
                "req-invalid",
            )
        assert getattr(exc_info.value, "code", None) == "PROVIDER_INVALID_REQUEST"
        assert primary.calls == 1
        assert fallback.calls == 0


@pytest.mark.anyio
async def test_structured_and_embedding_gateway_paths() -> None:
    from app.schemas.gateway import EmbeddingRequest, StructuredChatRequest

    primary = FakeProvider("primary")
    fallback = FakeProvider("fallback")
    with SessionLocal() as db:
        service = build_service(db, [primary, fallback])
        structured = await service.structured(
            StructuredChatRequest(
                messages=[Message(role="user", content="extract")],
                schema_name="answer",
                response_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            ),
            "req-structured",
        )
        assert structured.data == {"answer": "ok"}

        embedded = await service.embed(
            EmbeddingRequest(model="embedding", inputs=["hello", "world"]),
            "req-embed",
        )
        assert embedded.dimensions == 3
        assert len(embedded.embeddings) == 2
```

The first gateway test proves an important policy: a transient failure retries the current provider before moving to the fallback provider.

```python
assert result.fallback_used is False
assert len(result.attempts) == 2
assert fallback.calls == 0
```

If this test breaks after a refactor, the routing policy has changed.

## 34. Provider adapter normalization tests

**File: `tests/test_providers.py`**

```python
from types import SimpleNamespace

import pytest

from app.providers.base import (
    ProviderChatInput,
    ProviderEmbeddingInput,
    ProviderStructuredInput,
)
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider


class FakeGeminiInteractions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        structured = "response_format" in kwargs
        return SimpleNamespace(
            id="gemini-response-1",
            output_text='{"answer":"ok"}' if structured else "gemini reply",
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=4,
                total_tokens=16,
                cached_input_tokens=2,
            ),
        )


class FakeGeminiModels:
    def embed_content(self, **kwargs):
        assert kwargs["model"] == "gemini-embedding-001"
        return SimpleNamespace(
            id="gemini-embed-1",
            embeddings=[SimpleNamespace(values=[0.1, 0.2]), SimpleNamespace(values=[0.3, 0.4])],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=0,
                total_token_count=8,
                cached_content_token_count=0,
            ),
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.interactions = FakeGeminiInteractions()
        self.models = FakeGeminiModels()


class FakeOpenAIResponses:
    async def create(self, **kwargs):
        structured = "text" in kwargs
        return SimpleNamespace(
            id="openai-response-1",
            output_text='{"answer":"ok"}' if structured else "openai reply",
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=5,
                total_tokens=25,
                input_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
        )


class FakeOpenAIEmbeddings:
    async def create(self, **kwargs):
        assert kwargs["model"] == "text-embedding-3-small"
        return SimpleNamespace(
            id="openai-embed-1",
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])],
            usage=SimpleNamespace(prompt_tokens=9, total_tokens=9),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()
        self.embeddings = FakeOpenAIEmbeddings()


@pytest.mark.anyio
async def test_gemini_provider_normalizes_chat_structured_and_embeddings() -> None:
    provider = GeminiProvider("test-key")
    provider._client = FakeGeminiClient()

    chat = await provider.chat(
        ProviderChatInput(
            model="gemini-3.5-flash",
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
        )
    )
    assert chat.content == "gemini reply"
    assert chat.usage.total_tokens == 16
    assert chat.usage.cached_input_tokens == 2

    structured = await provider.structured(
        ProviderStructuredInput(
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "Extract"}],
            schema_name="answer",
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        )
    )
    assert structured.structured_data == {"answer": "ok"}

    embedded = await provider.embed(
        ProviderEmbeddingInput(
            model="gemini-embedding-001",
            inputs=["hello", "world"],
        )
    )
    assert embedded.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert embedded.usage.total_tokens == 8


@pytest.mark.anyio
async def test_openai_provider_normalizes_chat_structured_and_embeddings() -> None:
    provider = OpenAIProvider("test-key", 10)
    provider._client = FakeOpenAIClient()

    chat = await provider.chat(
        ProviderChatInput(
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
        )
    )
    assert chat.content == "openai reply"
    assert chat.usage.total_tokens == 25
    assert chat.usage.cached_input_tokens == 3

    structured = await provider.structured(
        ProviderStructuredInput(
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "Extract"}],
            schema_name="answer",
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
    )
    assert structured.structured_data == {"answer": "ok"}

    embedded = await provider.embed(
        ProviderEmbeddingInput(
            model="text-embedding-3-small",
            inputs=["hello", "world"],
        )
    )
    assert embedded.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert embedded.usage.total_tokens == 9


def test_provider_configuration_and_error_classification() -> None:
    gemini = GeminiProvider("")
    openai = OpenAIProvider("", 10)

    with pytest.raises(Exception) as gemini_exc:
        gemini._get_client()
    assert getattr(gemini_exc.value, "code", None) == "GEMINI_API_KEY_MISSING"

    with pytest.raises(Exception) as openai_exc:
        openai._get_client()
    assert getattr(openai_exc.value, "code", None) == "OPENAI_API_KEY_MISSING"

    gemini_rate_limit = gemini._classify(
        SimpleNamespace(status_code=429, __str__=lambda self: "rate")
    )
    assert gemini_rate_limit.retryable is True

    openai_bad_request = openai._classify(
        SimpleNamespace(status_code=400, __str__=lambda self: "bad")
    )
    assert openai_bad_request.retryable is False
```

These tests create fake SDK-shaped responses and inject fake clients directly into provider adapters. The goal is to test normalization without real API traffic.

```python
provider = GeminiProvider("test-key")
provider._client = FakeGeminiClient()
```

Directly assigning a private test client is acceptable inside focused adapter tests. Application code should not do this.

The adapter tests verify token-field normalization, structured JSON parsing, and embedding-vector extraction. These are the areas most likely to break when a provider SDK response shape changes.

## 35. Endpoint tests vs service tests

**File: `tests/test_gateway_endpoints.py`**

```python
import pytest
from httpx import AsyncClient

from app.api.v1.endpoints.gateway import get_gateway_service
from app.main import app
from app.schemas.gateway import (
    AttemptData,
    ChatData,
    EmbeddingData,
    StructuredChatData,
    TokenUsageData,
)


class FakeGatewayService:
    async def chat(self, request, request_id: str) -> ChatData:
        return ChatData(
            request_id=request_id,
            model_alias=request.model,
            provider="fake",
            model="fake-chat",
            content="fake reply",
            fallback_used=False,
            usage=TokenUsageData(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_input_tokens=0,
                estimated_cost_usd=0.001,
            ),
            attempts=[
                AttemptData(
                    provider="fake",
                    model="fake-chat",
                    attempt_number=1,
                    status="success",
                    latency_ms=1.2,
                )
            ],
        )

    async def structured(self, request, request_id: str) -> StructuredChatData:
        return StructuredChatData(
            request_id=request_id,
            model_alias=request.model,
            provider="fake",
            model="fake-structured",
            data={"answer": "ok"},
            fallback_used=False,
            usage=TokenUsageData(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_input_tokens=0,
                estimated_cost_usd=0.001,
            ),
            attempts=[
                AttemptData(
                    provider="fake",
                    model="fake-structured",
                    attempt_number=1,
                    status="success",
                    latency_ms=1.0,
                )
            ],
        )

    async def embed(self, request, request_id: str) -> EmbeddingData:
        return EmbeddingData(
            request_id=request_id,
            model_alias=request.model,
            provider="fake",
            model="fake-embed",
            dimensions=3,
            embeddings=[[0.1, 0.2, 0.3] for _ in request.inputs],
            fallback_used=False,
            usage=TokenUsageData(
                input_tokens=5,
                output_tokens=0,
                total_tokens=5,
                cached_input_tokens=0,
                estimated_cost_usd=0.0001,
            ),
            attempts=[
                AttemptData(
                    provider="fake",
                    model="fake-embed",
                    attempt_number=1,
                    status="success",
                    latency_ms=0.5,
                )
            ],
        )


@pytest.fixture(autouse=True)
def override_gateway() -> None:
    app.dependency_overrides[get_gateway_service] = lambda: FakeGatewayService()
    yield
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_chat_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"model": "fast", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["data"]["content"] == "fake reply"


@pytest.mark.anyio
async def test_structured_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/structured",
        json={
            "model": "fast",
            "messages": [{"role": "user", "content": "extract"}],
            "schema_name": "answer",
            "response_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["data"] == {"answer": "ok"}


@pytest.mark.anyio
async def test_embedding_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/embeddings",
        json={"model": "embedding", "inputs": ["hello", "world"]},
    )
    assert response.status_code == 200
    assert response.json()["data"]["dimensions"] == 3
    assert len(response.json()["data"]["embeddings"]) == 2


@pytest.mark.anyio
async def test_structured_schema_validation(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/structured",
        json={
            "messages": [{"role": "user", "content": "extract"}],
            "response_schema": {"type": "array", "items": {"type": "string"}},
        },
    )
    assert response.status_code == 422
```

Endpoint tests replace `get_gateway_service` with `FakeGatewayService`. They test HTTP concerns: path, body validation, serialization, status codes, and response shape.

Gateway service tests do the opposite. They call `GatewayService` directly with fake providers and test retries, fallback, and persistence behavior.

```text
Endpoint test: HTTP contract
Service test: orchestration policy
Provider test: SDK normalization
Routing test: target selection
```

This test layering gives failures a location. If endpoint tests fail but service tests pass, inspect API wiring. If routing tests fail, inspect alias policy. If provider normalization tests fail, inspect the adapter.

## 36. Docker persistence

**File: `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY .env.example .env.example

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**File: `compose.yaml`**

```yaml
services:
  gateway:
    build:
      context: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - gateway_data:/app/data
    environment:
      DATABASE_URL: sqlite:////app/data/gateway.db
    restart: unless-stopped

volumes:
  gateway_data:
```

### Dependency compatibility ranges

**Files: `requirements.txt` and `requirements-dev.txt`**

```text
fastapi[standard-no-fastapi-cloud-cli]>=0.128.2,<1.0
pydantic-settings>=2.14.1,<3.0
SQLAlchemy>=2.0.50,<3.0
anyio>=4.13.0,<5.0
google-genai>=2.11.0,<3.0
openai>=2.45.0,<3.0
```

```text
-r requirements.txt

httpx>=0.28.1,<1.0
pytest>=9.0.2,<10.0
pytest-cov>=7.0.0,<8.0
ruff>=0.15.21,<1.0
```

Project 1 used simple package names because dependency lifecycle was not the lesson. A provider gateway is more sensitive to third-party SDK changes: `google-genai` and `openai` are adapter boundaries whose method signatures and response objects matter to our normalization code.

The requirement syntax:

```text
>=2.11.0
```

means "do not install a version older than the one used as the compatibility floor."

```text
<3.0
```

means "do not silently cross into the next major version."

So:

```text
google-genai>=2.11.0,<3.0
```

means:

```text
compatible 2.x release at or above 2.11.0
```

It is not the same as an exact lock file. Exact pins maximize reproducibility for one environment; compatible ranges make a public starter easier to update. This project uses ranges plus Dependabot and CI: dependency updates should arrive as reviewable pull requests and must pass adapter tests before merge.

The final package was validated with `google-genai 2.11.0` and `openai 2.45.0`. The provider adapters are still tested with fake SDK-shaped responses because automated tests must not spend API credits or require secrets. A separate import/client-construction smoke check verified that the declared SDK packages can create the real Gemini and OpenAI client objects.

The Dockerfile itself uses local SQLite by default. Docker Compose overrides the database path to `/app/data/gateway.db` and mounts a named volume at `/app/data`.

```text
container replaced
    |
    v
named volume remains
    |
    v
usage history remains
```

Without the volume, SQLite data inside the container writable layer would disappear when the container is removed.

## 37. CI and CodeQL

**File: `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    name: Lint and Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-dev.txt
      - run: pip install -r requirements-dev.txt
      - run: ruff check .
      - run: ruff format --check .

  test:
    name: Tests and Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-dev.txt
      - run: pip install -r requirements-dev.txt
      - run: pytest --cov=app --cov-report=term-missing --cov-report=xml --cov-fail-under=80

  docker:
    name: Build Docker Image
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - run: docker build -t llm-api-gateway:ci .
```

**File: `.github/workflows/codeql.yml`**

```yaml
name: CodeQL

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: "17 3 * * 1"

permissions:
  contents: read
  security-events: write

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: github/codeql-action/init@v4
        with:
          languages: python
      - uses: github/codeql-action/analyze@v4
```

CI separates quality, behavior, and packaging checks into independent jobs. A green test job does not prove the Dockerfile builds. A green Docker job does not prove provider fallback behavior is correct.

CodeQL is static analysis. It does not replace tests. Tests check expected behavior; static analysis searches code patterns for security and correctness issues.

## 38. Complete request flow: chat

```text
1. POST /api/v1/chat
2. middleware chooses request_id
3. Pydantic validates ChatRequest
4. endpoint gets GatewayService
5. GatewayService validates message policy limits
6. RoutingService resolves alias + capability
7. GatewayService starts target/retry loops
8. provider adapter translates normalized request
9. provider SDK executes
10. adapter normalizes text + token usage
11. PricingCatalog estimates cost
12. UsageService persists attempt
13. GatewayService builds ChatData
14. endpoint wraps SuccessResponse
15. middleware adds response headers
16. client receives normalized gateway response
```

## 39. Complete failure flow: retry and fallback

```text
fast alias
  -> Gemini
      attempt 1 -> 429
      classify RATE_LIMIT / retryable
      persist failed attempt
      backoff + jitter
      attempt 2 -> 503
      classify TRANSIENT / retryable
      persist failed attempt
      retries exhausted
  -> OpenAI
      attempt 3 -> success
      normalize usage
      estimate cost
      persist success
      return response with fallback_used=true
```

This is the main interview story of Project 3.

## 40. Why not put retry logic inside every adapter?

Because then Gemini and OpenAI may have different application-level retry semantics by accident. A gateway should define one reliability policy and let adapters focus on protocol translation.

Provider SDKs may have their own internal retries. In this project's OpenAI adapter, SDK retries are disabled so gateway attempts remain explicit. Gemini's official SDK may perform transient retries internally; the notes and README treat gateway attempts as gateway-level attempts, not a perfect count of every lower-level HTTP transmission. In a stricter production system you would align or disable all SDK retry layers where supported.

## 41. Why fallback does not mean zero downtime

Fallback improves resilience. It does not guarantee equivalent behavior. Different models may differ in quality, context limits, structured-schema support, latency, content policies, and pricing.

A production gateway should evaluate fallback models against application-specific tests. The gateway solves routing mechanics, not semantic equivalence.

## 42. Why model aliases are not model rankings

`fast`, `cheap`, and `reasoning` are local product policy names. They do not claim that one vendor model is universally the fastest, cheapest, or best reasoning model in every region or every pricing tier.

The names describe how this repository intends to route workloads. You can change the chains in `.env` without changing API clients.

## 43. Cost estimation limitations

The gateway estimates cost from provider-reported token counts and a local price catalog. It is not an invoice reconciliation system.

Real billing can differ because of batch/flex/priority tiers, long-context pricing, regional processing, cached-token rules, tool use, multimodal units, promotional/free tiers, or provider-specific billing adjustments.

That is why the response field is named `estimated_cost_usd`.

## 44. Security boundaries before public deployment

```text
Current portfolio gateway
  /usage unauthenticated
  no tenant isolation
  no per-client rate limits
  provider keys stored in server env

Public production gateway needs
  authentication
  authorization
  client quotas
  request size controls
  tenant-aware usage accounting
  secret management
  abuse controls
  deployment threat model
```

Do not put provider keys in a frontend application. The point of the gateway is partly to keep provider credentials on the server.

## 45. Debugging map

```text
404 route missing
  -> app/api/v1/router.py
  -> app/main.py

MODEL_ALIAS_UNKNOWN
  -> .env alias name
  -> RoutingService._aliases

NO_PROVIDER_AVAILABLE
  -> provider API keys
  -> alias chain
  -> requested capability
  -> provider hint

ALL_PROVIDERS_FAILED
  -> response attempts[]
  -> usage_events rows
  -> provider error classification

Wrong cost
  -> provider token normalization
  -> config/pricing.json
  -> PricingCatalog

Retries not happening
  -> ProviderError.retryable
  -> ProviderErrorKind
  -> GATEWAY_MAX_RETRIES

Fallback not happening
  -> alias chain ordering
  -> provider configured state
  -> GatewayService outer target loop

Usage endpoint empty after real request
  -> DATABASE_URL
  -> UsageService.record
  -> UsageRepository.add
  -> Docker volume path
```

## 46. Commands you should run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
ruff check .
ruff format --check .
pytest --cov=app --cov-report=term-missing --cov-fail-under=80
uvicorn app.main:app --reload
```

Then open Swagger and test metadata first. `/providers` tells you whether your keys were loaded without making a billable model request. `/models` shows route chains. `/ready` proves the database and at least one provider configuration are usable.

## 47. Suggested real manual tests

```text
Test 1
GET /providers
Expected: Gemini configured if GEMINI_API_KEY is present

Test 2
POST /chat with model=fast
Expected: normalized text response

Test 3
POST /chat/structured
Expected: data follows your JSON Schema

Test 4
POST /embeddings
Expected: vector count equals input count

Test 5
GET /usage
Expected: provider attempts and token totals increased

Test 6
Send X-Request-ID: manual-trace-1
Expected: same ID in header, response body, and usage event

Test 7
Temporarily put an invalid first route before a valid configured fallback
Expected: inspect whether error classification allows fallback for that failure type
```

## 48. Interview questions and answers

### 48.1 Why use an LLM gateway?

To centralize provider abstraction, model routing, reliability policy, usage normalization, and cost observability so application code is not coupled to vendor SDKs.

### 48.2 What is a provider adapter?

A class that translates a normalized internal request into provider-specific SDK calls and translates provider-specific responses/errors back into normalized gateway contracts.

### 48.3 What is a model alias?

A stable application-level intent name such as fast or reasoning that maps to an ordered provider/model route chain.

### 48.4 Retry vs fallback?

Retry repeats the same route after a retryable transient failure. Fallback moves to another provider/model route after retries are exhausted or a route is unusable.

### 48.5 Why exponential backoff?

It reduces immediate repeated pressure on an overloaded provider by increasing the delay between retries.

### 48.6 Why jitter?

It prevents many clients or replicas from retrying at exactly the same time after a shared failure.

### 48.7 Why fail fast on invalid requests?

The input is the problem; retrying or switching providers wastes latency and cost and may hide the caller bug.

### 48.8 Why normalize token usage?

Provider SDKs expose different metadata fields. The gateway needs one usage contract for cost calculation, persistence, and API responses.

### 48.9 Why record failed attempts?

Fallback behavior and provider reliability are invisible if only the final successful response is stored.

### 48.10 Why separate pricing from adapters?

Adapters should normalize protocol data. Pricing is changing operational configuration and should be updateable without rewriting provider SDK logic.

### 48.11 Why request IDs?

They correlate one logical request across HTTP responses, logs, provider attempts, and usage records.

### 48.12 Why capability filtering?

A configured provider is not automatically valid for every operation; routing must ensure the adapter supports chat, structured output, or embeddings.

### 48.13 Why a provider registry?

It replaces provider-name conditionals with a lookup boundary and lets routing work with the common provider interface.

### 48.14 Why disable OpenAI SDK retries here?

To make gateway-level attempts explicit and avoid hidden multiplication of retry policies.

### 48.15 What is the repository pattern used for?

To isolate SQL query mechanics from business/orchestration services.

### 48.16 Is estimated cost exact billing?

No. It is based on provider-reported tokens and a dated local catalog. Billing tiers and provider rules can differ.

### 48.17 Does fallback guarantee equivalent answers?

No. It improves availability mechanics, but semantic quality and model behavior must be evaluated separately.

### 48.18 Why not expose raw model names to app clients?

Aliases reduce coupling and let routing policy change without forcing every client to change.

### 48.19 What would you add before public deployment?

Authentication, authorization, client quotas/rate limiting, tenant-aware accounting, secret management, stronger observability, and a threat model.

### 48.20 What is the one-sentence project explanation?

I built a provider-neutral FastAPI LLM gateway that routes intent aliases across Gemini and OpenAI with centralized retries, fallback, timeout handling, normalized usage, cost estimation, request tracing, structured output, embeddings, and persistent attempt analytics.

## 49. What not to add before moving to Project 4

Do not add LangChain, agents, vector databases, Kafka, Kubernetes, Redis, or a frontend just to make the repository larger. Project 4 is the RAG project. Embeddings from this gateway will become useful there. Keep Project 3 focused on the LLM infrastructure boundary.

## 50. Final mental model

```text
APPLICATION INTENT
      |
      v
  MODEL ALIAS
      |
      v
 ROUTING POLICY
      |
      v
ORDERED TARGETS
      |
      v
 RETRY SAME ROUTE
      |
      +---- failure exhausted ----> FALLBACK NEXT ROUTE
      |
      v
PROVIDER ADAPTER
      |
      v
PROVIDER SDK / API
      |
      v
NORMALIZED RESULT
      |
      +--> token usage
      +--> estimated cost
      +--> attempt persistence
      +--> request correlation
      |
      v
STABLE GATEWAY RESPONSE
```

If you understand that diagram and can explain where each responsibility lives, you understand the architecture of Project 3.

## Appendix A — Route reference

`GET /api/v1/health` — liveness.

`GET /api/v1/ready` — database/provider readiness.

`GET /api/v1/providers` — provider metadata.

`GET /api/v1/models` — alias metadata.

`POST /api/v1/chat` — normalized chat.

`POST /api/v1/chat/structured` — schema-constrained JSON.

`POST /api/v1/embeddings` — text vectors.

`GET /api/v1/usage` — attempt/token/cost analytics.

## Appendix B — Environment variable map

**File: `.env.example`**

```env
APP_NAME=LLM API Gateway
APP_ENV=local
APP_DEBUG=true
API_V1_PREFIX=/api/v1
LOG_LEVEL=INFO

DATABASE_URL=sqlite:///./gateway.db
PRICING_CATALOG_PATH=config/pricing.json

GATEWAY_TIMEOUT_SECONDS=45
GATEWAY_MAX_RETRIES=2
GATEWAY_BACKOFF_BASE_SECONDS=0.25
GATEWAY_BACKOFF_MAX_SECONDS=4

MODEL_ALIAS_FAST=gemini:gemini-3.5-flash,openai:gpt-5.6-terra
MODEL_ALIAS_REASONING=openai:gpt-5.6-sol,gemini:gemini-3.5-flash
MODEL_ALIAS_CHEAP=gemini:gemini-3.1-flash-lite,openai:gpt-5.6-luna
MODEL_ALIAS_EMBEDDING=gemini:gemini-embedding-001,openai:text-embedding-3-small

GEMINI_API_KEY=
OPENAI_API_KEY=

MAX_MESSAGES=50
MAX_MESSAGE_CHARS=20000
MAX_EMBEDDING_INPUTS=100
MAX_EMBEDDING_CHARS=8000
```

When debugging configuration, remember that `get_settings()` is cached. In a running Python process, changing `.env` does not rebuild the cached Settings object automatically. Restart Uvicorn after changing environment variables.

## Appendix C — Public scope checklist

```text
Provider abstraction             [x]
Gemini adapter                  [x]
OpenAI adapter                  [x]
Alias routing                   [x]
Provider hint                   [x]
Capability filtering            [x]
Timeout boundary                [x]
Retry classification            [x]
Exponential backoff             [x]
Jitter                          [x]
Fallback routing                [x]
Normalized errors               [x]
Normalized token usage          [x]
Cost estimation                 [x]
Pricing catalog                 [x]
Request IDs                     [x]
Latency metadata                [x]
Usage persistence               [x]
Usage analytics                 [x]
Structured output               [x]
Embeddings                      [x]
Tests without paid API calls    [x]
Coverage floor                  [x]
Ruff                           [x]
Docker                          [x]
Persistent Compose volume       [x]
CI                              [x]
CodeQL                          [x]
Dependabot                      [x]
README                          [x]
Detailed NOTES                  [x]
```
