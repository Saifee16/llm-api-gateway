from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default="fast", min_length=1, max_length=64)
    messages: list[Message] = Field(min_length=1)
    provider: str | None = Field(default=None, max_length=32)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1, le=100_000)


class StructuredChatRequest(ChatRequest):
    schema_name: str = Field(default="response", pattern=r"^[A-Za-z0-9_-]+$")
    response_schema: dict[str, Any]

    @field_validator("response_schema")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("response_schema must have top-level type='object'")
        if "properties" not in value:
            raise ValueError("response_schema must define properties")
        return value


class EmbeddingRequest(BaseModel):
    model: str = Field(default="embedding", min_length=1, max_length=64)
    inputs: list[str] = Field(min_length=1)
    provider: str | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_inputs(self) -> "EmbeddingRequest":
        if any(not item.strip() for item in self.inputs):
            raise ValueError("embedding inputs cannot be blank")
        return self


class TokenUsageData(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float | None


class AttemptData(BaseModel):
    provider: str
    model: str
    attempt_number: int
    status: Literal["success", "failed"]
    latency_ms: float
    error_code: str | None = None


class ChatData(BaseModel):
    request_id: str
    model_alias: str
    provider: str
    model: str
    content: str
    fallback_used: bool
    usage: TokenUsageData
    attempts: list[AttemptData]


class StructuredChatData(BaseModel):
    request_id: str
    model_alias: str
    provider: str
    model: str
    data: dict[str, Any]
    fallback_used: bool
    usage: TokenUsageData
    attempts: list[AttemptData]


class EmbeddingData(BaseModel):
    request_id: str
    model_alias: str
    provider: str
    model: str
    dimensions: int
    embeddings: list[list[float]]
    fallback_used: bool
    usage: TokenUsageData
    attempts: list[AttemptData]
