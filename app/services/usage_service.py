from datetime import UTC, datetime, timedelta

from app.db.models import UsageEvent
from app.providers.base import ProviderUsage
from app.repositories.usage_repository import UsageRepository
from app.schemas.usage import (
    UsageBreakdownData,
    UsageData,
    UsageEventData,
    UsageTotalsData,
)


class UsageService:
    def __init__(self, repository: UsageRepository) -> None:
        self.repository = repository

    def record(
        self,
        *,
        request_id: str,
        operation: str,
        model_alias: str,
        provider: str,
        model: str,
        attempt_number: int,
        success: bool,
        error_code: str | None,
        latency_ms: float,
        usage: ProviderUsage,
        estimated_cost_usd: float | None,
    ) -> None:
        self.repository.add(
            UsageEvent(
                request_id=request_id,
                operation=operation,
                model_alias=model_alias,
                provider=provider,
                model=model,
                attempt_number=attempt_number,
                success=success,
                error_code=error_code,
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
        )

    def summary(self, *, days: int, limit: int) -> UsageData:
        since = datetime.now(UTC) - timedelta(days=days)
        totals = self.repository.totals(since=since)
        recent = self.repository.recent(since=since, limit=limit)
        breakdown = self.repository.breakdown(since=since)
        successful = int(totals["successful_attempts"])
        attempts = int(totals["attempts"])

        return UsageData(
            window_days=days,
            totals=UsageTotalsData(
                attempts=attempts,
                successful_attempts=successful,
                failed_attempts=attempts - successful,
                input_tokens=int(totals["input_tokens"]),
                output_tokens=int(totals["output_tokens"]),
                total_tokens=int(totals["total_tokens"]),
                estimated_cost_usd=round(float(totals["estimated_cost_usd"]), 10),
            ),
            breakdown=[UsageBreakdownData(**item) for item in breakdown],
            recent_events=[
                UsageEventData(
                    request_id=item.request_id,
                    operation=item.operation,
                    model_alias=item.model_alias,
                    provider=item.provider,
                    model=item.model,
                    attempt_number=item.attempt_number,
                    success=item.success,
                    error_code=item.error_code,
                    latency_ms=item.latency_ms,
                    total_tokens=item.total_tokens,
                    estimated_cost_usd=item.estimated_cost_usd,
                    created_at=item.created_at,
                )
                for item in recent
            ],
        )
