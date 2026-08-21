from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ProviderCapability(StrEnum):
    CHAT = "chat"
    STRUCTURED = "structured"
    EMBEDDINGS = "embeddings"


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass(frozen=True)
class ProviderChatInput:
    model: str
    messages: list[dict[str, str]]
    temperature: float | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ProviderStructuredInput(ProviderChatInput):
    schema_name: str = "response"
    response_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderEmbeddingInput:
    model: str
    inputs: list[str]


@dataclass(frozen=True)
class ProviderChatResult:
    content: str
    usage: ProviderUsage
    provider_request_id: str | None = None
    structured_data: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderEmbeddingResult:
    embeddings: list[list[float]]
    usage: ProviderUsage
    provider_request_id: str | None = None


class LLMProvider(ABC):
    name: str
    capabilities: frozenset[ProviderCapability]

    @property
    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    def supports(self, capability: ProviderCapability) -> bool:
        return capability in self.capabilities

    @abstractmethod
    async def chat(self, request: ProviderChatInput) -> ProviderChatResult:
        raise NotImplementedError

    @abstractmethod
    async def structured(self, request: ProviderStructuredInput) -> ProviderChatResult:
        raise NotImplementedError

    @abstractmethod
    async def embed(self, request: ProviderEmbeddingInput) -> ProviderEmbeddingResult:
        raise NotImplementedError
