import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
from app.providers.base import ProviderCapability
from app.providers.registry import ProviderRegistry
from app.services.routing_service import RoutingService
from tests.fakes import FakeProvider


def make_service() -> RoutingService:
    settings = Settings(
        MODEL_ALIAS_FAST="primary:model-a,fallback:model-b",
        MODEL_ALIAS_REASONING="primary:model-r",
        MODEL_ALIAS_CHEAP="primary:model-c",
        MODEL_ALIAS_EMBEDDING="fallback:embed-b",
    )
    registry = ProviderRegistry([FakeProvider("primary"), FakeProvider("fallback")])
    return RoutingService(settings, registry)


def test_routing_preserves_fallback_order() -> None:
    service = make_service()
    targets = service.resolve(alias="fast", capability=ProviderCapability.CHAT)
    assert [(item.provider, item.model) for item in targets] == [
        ("primary", "model-a"),
        ("fallback", "model-b"),
    ]


def test_provider_hint_filters_routes() -> None:
    service = make_service()
    targets = service.resolve(
        alias="fast",
        capability=ProviderCapability.CHAT,
        provider_hint="fallback",
    )
    assert len(targets) == 1
    assert targets[0].provider == "fallback"


def test_unknown_alias_is_clean_app_error() -> None:
    service = make_service()
    with pytest.raises(AppError) as exc_info:
        service.resolve(alias="missing", capability=ProviderCapability.CHAT)
    assert exc_info.value.code == "MODEL_ALIAS_UNKNOWN"
