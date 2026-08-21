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
    capabilities = frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.STRUCTURED,
            ProviderCapability.EMBEDDINGS,
        }
    )

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
