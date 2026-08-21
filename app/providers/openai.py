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
    capabilities = frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.STRUCTURED,
            ProviderCapability.EMBEDDINGS,
        }
    )

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
