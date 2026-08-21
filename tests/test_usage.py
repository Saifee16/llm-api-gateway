from datetime import UTC, datetime

import pytest
from httpx import AsyncClient

from app.db.database import SessionLocal
from app.db.models import UsageEvent
from app.repositories.usage_repository import UsageRepository
from app.services.usage_service import UsageService


def test_usage_service_summarizes_success_and_failure() -> None:
    with SessionLocal() as db:
        repository = UsageRepository(db)
        now = datetime.now(UTC)
        repository.add(
            UsageEvent(
                request_id="r1",
                operation="chat",
                model_alias="fast",
                provider="gemini",
                model="gemini-3.5-flash",
                attempt_number=1,
                success=True,
                error_code=None,
                latency_ms=10,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cached_input_tokens=0,
                estimated_cost_usd=0.001,
                created_at=now,
            )
        )
        repository.add(
            UsageEvent(
                request_id="r2",
                operation="chat",
                model_alias="fast",
                provider="gemini",
                model="gemini-3.5-flash",
                attempt_number=1,
                success=False,
                error_code="RATE_LIMIT",
                latency_ms=5,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cached_input_tokens=0,
                estimated_cost_usd=None,
                created_at=now,
            )
        )
        summary = UsageService(repository).summary(days=7, limit=10)
        assert summary.totals.attempts == 2
        assert summary.totals.successful_attempts == 1
        assert summary.totals.failed_attempts == 1
        assert summary.totals.total_tokens == 150
        assert summary.totals.estimated_cost_usd == 0.001


@pytest.mark.anyio
async def test_usage_endpoint_empty_database(client: AsyncClient) -> None:
    response = await client.get("/api/v1/usage?days=7&limit=10")
    assert response.status_code == 200
    assert response.json()["data"]["totals"]["attempts"] == 0
