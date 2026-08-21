from fastapi import APIRouter

from app.api.v1.endpoints import gateway, health, metadata, usage

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(metadata.router, tags=["Gateway Metadata"])
api_router.include_router(gateway.router, tags=["Gateway"])
api_router.include_router(usage.router, tags=["Usage"])
