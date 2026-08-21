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
