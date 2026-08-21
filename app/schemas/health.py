from typing import Literal

from pydantic import BaseModel


class HealthData(BaseModel):
    status: Literal["ok"] = "ok"
    app: str
    environment: str


class ReadyData(BaseModel):
    status: Literal["ready"] = "ready"
    database: Literal["ok"] = "ok"
    configured_providers: int
