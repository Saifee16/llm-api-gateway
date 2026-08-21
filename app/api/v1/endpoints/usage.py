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
