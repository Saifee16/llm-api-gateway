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
