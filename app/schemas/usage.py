from datetime import datetime

from pydantic import BaseModel


class UsageTotalsData(BaseModel):
    attempts: int
    successful_attempts: int
    failed_attempts: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class UsageBreakdownData(BaseModel):
    provider: str
    model: str
    attempts: int
    total_tokens: int
    estimated_cost_usd: float


class UsageEventData(BaseModel):
    request_id: str
    operation: str
    model_alias: str
    provider: str
    model: str
    attempt_number: int
    success: bool
    error_code: str | None
    latency_ms: float
    total_tokens: int
    estimated_cost_usd: float | None
    created_at: datetime


class UsageData(BaseModel):
    window_days: int
    totals: UsageTotalsData
    breakdown: list[UsageBreakdownData]
    recent_events: list[UsageEventData]
