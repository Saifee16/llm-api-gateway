import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_health_and_request_headers(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "req-known"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-known"
    assert "X-Response-Time-Ms" in response.headers
    body = response.json()
    assert body["success"] is True
    assert body["request_id"] == "req-known"
    assert body["data"]["status"] == "ok"


@pytest.mark.anyio
async def test_invalid_request_id_is_replaced_and_large_request_is_rejected(
    client: AsyncClient,
) -> None:
    response = await client.get("/api/v1/health", headers={"X-Request-ID": "invalid request id"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "invalid request id"
    assert len(response.headers["X-Request-ID"]) == 36

    response = await client.post(
        "/api/v1/chat",
        content="x" * 1_000_001,
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"


@pytest.mark.anyio
async def test_ready_has_configured_provider(client: AsyncClient) -> None:
    response = await client.get("/api/v1/ready")
    assert response.status_code == 200
    assert response.json()["data"]["configured_providers"] >= 1
