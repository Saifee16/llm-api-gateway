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
