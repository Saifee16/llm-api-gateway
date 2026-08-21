from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import UsageEvent


class UsageRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def add(self, event: UsageEvent) -> UsageEvent:
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def recent(self, *, since: datetime, limit: int) -> list[UsageEvent]:
        statement = (
            select(UsageEvent)
            .where(UsageEvent.created_at >= since)
            .order_by(UsageEvent.created_at.desc())
            .limit(limit)
        )
        return list(self.db.scalars(statement))

    def totals(self, *, since: datetime) -> dict[str, int | float]:
        statement = select(
            func.count(UsageEvent.id),
            func.sum(UsageEvent.input_tokens),
            func.sum(UsageEvent.output_tokens),
            func.sum(UsageEvent.total_tokens),
            func.sum(UsageEvent.estimated_cost_usd),
            func.sum(func.cast(UsageEvent.success, UsageEvent.id.type)),
        ).where(UsageEvent.created_at >= since)
        row = self.db.execute(statement).one()
        return {
            "attempts": int(row[0] or 0),
            "input_tokens": int(row[1] or 0),
            "output_tokens": int(row[2] or 0),
            "total_tokens": int(row[3] or 0),
            "estimated_cost_usd": float(row[4] or 0.0),
            "successful_attempts": int(row[5] or 0),
        }

    def breakdown(self, *, since: datetime) -> list[dict[str, object]]:
        statement = (
            select(
                UsageEvent.provider,
                UsageEvent.model,
                func.count(UsageEvent.id),
                func.sum(UsageEvent.total_tokens),
                func.sum(UsageEvent.estimated_cost_usd),
            )
            .where(UsageEvent.created_at >= since)
            .group_by(UsageEvent.provider, UsageEvent.model)
            .order_by(UsageEvent.provider, UsageEvent.model)
        )
        rows = self.db.execute(statement).all()
        return [
            {
                "provider": row[0],
                "model": row[1],
                "attempts": int(row[2] or 0),
                "total_tokens": int(row[3] or 0),
                "estimated_cost_usd": float(row[4] or 0.0),
            }
            for row in rows
        ]
