from collections import deque

from app.providers.base import (
    LLMProvider,
    ProviderCapability,
    ProviderChatInput,
    ProviderChatResult,
    ProviderEmbeddingInput,
    ProviderEmbeddingResult,
    ProviderStructuredInput,
    ProviderUsage,
)


class FakeProvider(LLMProvider):
    capabilities = frozenset(
        {
            ProviderCapability.CHAT,
            ProviderCapability.STRUCTURED,
            ProviderCapability.EMBEDDINGS,
        }
    )

    def __init__(
        self, name: str, outcomes: list[object] | None = None, configured: bool = True
    ) -> None:
        self.name = name
        self._configured = configured
        self.outcomes = deque(outcomes or [])
        self.calls = 0

    @property
    def configured(self) -> bool:
        return self._configured

    def _next(self, default: object) -> object:
        self.calls += 1
        if self.outcomes:
            value = self.outcomes.popleft()
            if isinstance(value, Exception):
                raise value
            return value
        return default

    async def chat(self, request: ProviderChatInput) -> ProviderChatResult:
        del request
        return self._next(
            ProviderChatResult(
                content=f"reply from {self.name}",
                usage=ProviderUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            )
        )

    async def structured(self, request: ProviderStructuredInput) -> ProviderChatResult:
        del request
        return self._next(
            ProviderChatResult(
                content='{"answer":"ok"}',
                structured_data={"answer": "ok"},
                usage=ProviderUsage(input_tokens=12, output_tokens=4, total_tokens=16),
            )
        )

    async def embed(self, request: ProviderEmbeddingInput) -> ProviderEmbeddingResult:
        return self._next(
            ProviderEmbeddingResult(
                embeddings=[[0.1, 0.2, 0.3] for _ in request.inputs],
                usage=ProviderUsage(input_tokens=8, total_tokens=8),
            )
        )
