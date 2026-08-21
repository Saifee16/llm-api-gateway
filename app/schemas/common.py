from typing import Literal, TypeVar

from pydantic import BaseModel

DataT = TypeVar("DataT")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, object] | None = None


class SuccessResponse[DataT](BaseModel):
    success: Literal[True] = True
    request_id: str | None = None
    data: DataT


class ErrorResponse(BaseModel):
    success: Literal[False] = False
    request_id: str | None = None
    error: ErrorDetail
