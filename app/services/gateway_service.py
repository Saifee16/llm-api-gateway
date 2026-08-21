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
                    # Retry jitter is scheduling only, not a security decision.
                    jitter = random.uniform(0, delay * 0.25) if delay > 0 else 0.0  # nosec B311
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
            if provider is None:
                raise RuntimeError(f"Resolved provider is no longer registered: {target.provider}")
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
        if not isinstance(result, ProviderChatResult):
            raise RuntimeError("Chat operation returned an invalid provider result.")
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
            if provider is None:
                raise RuntimeError(f"Resolved provider is no longer registered: {target.provider}")
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
        if not isinstance(result, ProviderChatResult):
            raise RuntimeError("Structured operation returned an invalid provider result.")
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
            if provider is None:
                raise RuntimeError(f"Resolved provider is no longer registered: {target.provider}")
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
        if not isinstance(result, ProviderEmbeddingResult):
            raise RuntimeError("Embedding operation returned an invalid provider result.")
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
