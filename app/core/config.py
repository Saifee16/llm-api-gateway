from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="LLM API Gateway", alias="APP_NAME")
    app_env: Literal["local", "development", "staging", "production"] = Field(
        default="local",
        alias="APP_ENV",
    )
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        alias="LOG_LEVEL",
    )

    database_url: str = Field(default="sqlite:///./gateway.db", alias="DATABASE_URL")
    pricing_catalog_path: Path = Field(
        default=Path("config/pricing.json"),
        alias="PRICING_CATALOG_PATH",
    )

    gateway_timeout_seconds: float = Field(default=45.0, alias="GATEWAY_TIMEOUT_SECONDS")
    gateway_max_retries: int = Field(default=2, ge=0, le=8, alias="GATEWAY_MAX_RETRIES")
    gateway_backoff_base_seconds: float = Field(
        default=0.25,
        ge=0.0,
        alias="GATEWAY_BACKOFF_BASE_SECONDS",
    )
    gateway_backoff_max_seconds: float = Field(
        default=4.0,
        ge=0.0,
        alias="GATEWAY_BACKOFF_MAX_SECONDS",
    )
    gateway_api_key: str = Field(default="", alias="GATEWAY_API_KEY")
    max_request_bytes: int = Field(
        default=1_000_000,
        ge=1_024,
        le=10_000_000,
        alias="MAX_REQUEST_BYTES",
    )

    model_alias_fast: str = Field(
        default="gemini:gemini-3.5-flash,openai:gpt-5.6-terra",
        alias="MODEL_ALIAS_FAST",
    )
    model_alias_reasoning: str = Field(
        default="openai:gpt-5.6-sol,gemini:gemini-3.5-flash",
        alias="MODEL_ALIAS_REASONING",
    )
    model_alias_cheap: str = Field(
        default="gemini:gemini-3.1-flash-lite,openai:gpt-5.6-luna",
        alias="MODEL_ALIAS_CHEAP",
    )
    model_alias_embedding: str = Field(
        default="gemini:gemini-embedding-001,openai:text-embedding-3-small",
        alias="MODEL_ALIAS_EMBEDDING",
    )

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    max_messages: int = Field(default=50, ge=1, le=500, alias="MAX_MESSAGES")
    max_message_chars: int = Field(default=20_000, ge=100, alias="MAX_MESSAGE_CHARS")
    max_embedding_inputs: int = Field(default=100, ge=1, alias="MAX_EMBEDDING_INPUTS")
    max_embedding_chars: int = Field(default=8_000, ge=100, alias="MAX_EMBEDDING_CHARS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_production_security(self) -> "Settings":
        if self.app_env == "production" and not self.gateway_api_key:
            raise ValueError("GATEWAY_API_KEY must be configured when APP_ENV=production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
