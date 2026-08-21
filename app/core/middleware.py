import re
import secrets
from time import perf_counter
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, settings: Settings) -> None:
        super().__init__(app)
        self.settings = settings

    @staticmethod
    def _request_id(candidate: str | None) -> str:
        if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
        return str(uuid4())

    @staticmethod
    def _error_response(request_id: str, status_code: int, code: str, message: str) -> JSONResponse:
        response = JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "request_id": request_id,
                "error": {"code": code, "message": message},
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = self._request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started = perf_counter()

        if self.settings.gateway_api_key and request.url.path not in {
            "/api/v1/health",
            "/api/v1/ready",
        }:
            supplied_key = request.headers.get("X-API-Key", "")
            if not secrets.compare_digest(supplied_key, self.settings.gateway_api_key):
                return self._error_response(
                    request_id,
                    401,
                    "GATEWAY_AUTH_REQUIRED",
                    "A valid gateway API key is required.",
                )

        response = await call_next(request)

        latency_ms = round((perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(latency_ms)

        logger.info(
            "%s %s -> %s | %.2fms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
            request_id,
        )
        return response
