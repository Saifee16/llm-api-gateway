from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncByteStream, AsyncClient

from app.core.body_limit import RequestBodyLimitMiddleware
from app.core.config import Settings
from app.core.middleware import RequestContextMiddleware


class ChunkedBody(AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def body_limit_app() -> FastAPI:
    settings = Settings(MAX_REQUEST_BYTES=1_024)
    app = FastAPI()
    app.add_middleware(RequestBodyLimitMiddleware, max_request_bytes=settings.max_request_bytes)
    app.add_middleware(RequestContextMiddleware, settings=settings)

    @app.post("/echo")
    async def echo(request: Request) -> dict[str, int]:
        return {"size": len(await request.body())}

    return app


@pytest.mark.anyio
async def test_request_body_limit_enforces_declared_and_actual_size(
    body_limit_app: FastAPI,
) -> None:
    transport = ASGITransport(app=body_limit_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        normal = await client.post("/echo", content=b"x" * 1_024)
        declared_oversize = await client.post("/echo", content=b"x" * 1_025)
        no_content_length = await client.post(
            "/echo", content=ChunkedBody([b"x" * 512, b"x" * 513])
        )
        false_content_length = await client.post(
            "/echo",
            content=ChunkedBody([b"x" * 512, b"x" * 513]),
            headers={"content-length": "1"},
        )

    assert normal.status_code == 200
    assert normal.json() == {"size": 1_024}
    for response in (declared_oversize, no_content_length, false_content_length):
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
