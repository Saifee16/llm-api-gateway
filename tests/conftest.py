import os

os.environ.setdefault("APP_ENV", "local")
os.environ.setdefault("DATABASE_URL", "sqlite://")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GATEWAY_BACKOFF_BASE_SECONDS", "0")
os.environ.setdefault("GATEWAY_BACKOFF_MAX_SECONDS", "0")

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.database import Base, engine
from app.main import app


@pytest.fixture(autouse=True)
def reset_database() -> None:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
