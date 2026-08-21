import pytest
from httpx import AsyncClient

from app.api.v1.endpoints.gateway import get_gateway_service
from app.main import app
from app.schemas.gateway import (
    AttemptData,
    ChatData,
    EmbeddingData,
    StructuredChatData,
    TokenUsageData,
)


class FakeGatewayService:
    async def chat(self, request, request_id: str) -> ChatData:
        return ChatData(
            request_id=request_id,
            model_alias=request.model,
            provider="fake",
            model="fake-chat",
            content="fake reply",
            fallback_used=False,
            usage=TokenUsageData(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_input_tokens=0,
                estimated_cost_usd=0.001,
            ),
            attempts=[
                AttemptData(
                    provider="fake",
                    model="fake-chat",
                    attempt_number=1,
                    status="success",
                    latency_ms=1.2,
                )
            ],
        )

    async def structured(self, request, request_id: str) -> StructuredChatData:
        return StructuredChatData(
            request_id=request_id,
            model_alias=request.model,
            provider="fake",
            model="fake-structured",
            data={"answer": "ok"},
            fallback_used=False,
            usage=TokenUsageData(
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cached_input_tokens=0,
                estimated_cost_usd=0.001,
            ),
            attempts=[
                AttemptData(
                    provider="fake",
                    model="fake-structured",
                    attempt_number=1,
                    status="success",
                    latency_ms=1.0,
                )
            ],
        )

    async def embed(self, request, request_id: str) -> EmbeddingData:
        return EmbeddingData(
            request_id=request_id,
            model_alias=request.model,
            provider="fake",
            model="fake-embed",
            dimensions=3,
            embeddings=[[0.1, 0.2, 0.3] for _ in request.inputs],
            fallback_used=False,
            usage=TokenUsageData(
                input_tokens=5,
                output_tokens=0,
                total_tokens=5,
                cached_input_tokens=0,
                estimated_cost_usd=0.0001,
            ),
            attempts=[
                AttemptData(
                    provider="fake",
                    model="fake-embed",
                    attempt_number=1,
                    status="success",
                    latency_ms=0.5,
                )
            ],
        )


@pytest.fixture(autouse=True)
def override_gateway() -> None:
    app.dependency_overrides[get_gateway_service] = lambda: FakeGatewayService()
    yield
    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_chat_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat",
        json={"model": "fast", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["data"]["content"] == "fake reply"


@pytest.mark.anyio
async def test_structured_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/structured",
        json={
            "model": "fast",
            "messages": [{"role": "user", "content": "extract"}],
            "schema_name": "answer",
            "response_schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["data"] == {"answer": "ok"}


@pytest.mark.anyio
async def test_embedding_endpoint(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/embeddings",
        json={"model": "embedding", "inputs": ["hello", "world"]},
    )
    assert response.status_code == 200
    assert response.json()["data"]["dimensions"] == 3
    assert len(response.json()["data"]["embeddings"]) == 2


@pytest.mark.anyio
async def test_structured_schema_validation(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/structured",
        json={
            "messages": [{"role": "user", "content": "extract"}],
            "response_schema": {"type": "array", "items": {"type": "string"}},
        },
    )
    assert response.status_code == 422
