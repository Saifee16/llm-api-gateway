import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from app.core.config import Settings
from app.core.middleware import RequestContextMiddleware


def test_production_requires_gateway_key() -> None:
    with pytest.raises(ValidationError, match="GATEWAY_API_KEY"):
        Settings(APP_ENV="production", GATEWAY_API_KEY="")

    settings = Settings(APP_ENV="production", GATEWAY_API_KEY="test-gateway-key")
    assert settings.gateway_api_key == "test-gateway-key"


@pytest.mark.anyio
async def test_gateway_key_protects_non_health_routes() -> None:
    app = FastAPI()
    app.add_middleware(
        RequestContextMiddleware,
        settings=Settings(GATEWAY_API_KEY="test-gateway-key"),
    )

    @app.get("/api/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/protected")
    async def protected() -> dict[str, str]:
        return {"status": "ok"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/api/v1/health")
        rejected = await client.get("/api/v1/protected")
        accepted = await client.get(
            "/api/v1/protected",
            headers={"X-API-Key": "test-gateway-key"},
        )

    assert health.status_code == 200
    assert rejected.status_code == 401
    assert rejected.json()["error"]["code"] == "GATEWAY_AUTH_REQUIRED"
    assert accepted.status_code == 200
