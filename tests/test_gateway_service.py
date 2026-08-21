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
