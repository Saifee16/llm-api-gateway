from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.body_limit import RequestBodyLimitMiddleware
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestContextMiddleware
from app.db.database import init_db

settings = get_settings()
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    del app
    init_db()
    logger.info("Database initialized")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        debug=settings.app_debug,
        lifespan=lifespan,
        description=(
            "Provider-neutral LLM gateway with alias routing, retries, fallback, "
            "normalized usage, cost estimation, structured output, and embeddings."
        ),
    )
    app.add_middleware(RequestBodyLimitMiddleware, max_request_bytes=settings.max_request_bytes)
    app.add_middleware(RequestContextMiddleware, settings=settings)
    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)
    return app


app = create_app()
