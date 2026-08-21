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
