import pytest
from httpx import AsyncClient


@pytest.mark.anyio
async def test_providers_endpoint(client: AsyncClient) -> None:
    response = await client.get("/api/v1/providers")
    assert response.status_code == 200
    providers = {row["name"]: row for row in response.json()["data"]["providers"]}
    assert providers["gemini"]["configured"] is True
    assert "structured" in providers["gemini"]["capabilities"]
    assert providers["openai"]["configured"] is False


@pytest.mark.anyio
async def test_models_endpoint_exposes_alias_chains(client: AsyncClient) -> None:
    response = await client.get("/api/v1/models")
    assert response.status_code == 200
    aliases = {row["alias"]: row for row in response.json()["data"]["aliases"]}
    assert set(aliases) == {"fast", "reasoning", "cheap", "embedding"}
    assert aliases["fast"]["targets"][0]["provider"] == "gemini"
