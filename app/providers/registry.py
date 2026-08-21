from functools import lru_cache

from app.core.config import get_settings
from app.providers.base import LLMProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider


class ProviderRegistry:
    def __init__(self, providers: list[LLMProvider]) -> None:
        self._providers = {provider.name: provider for provider in providers}

    def get(self, name: str) -> LLMProvider | None:
        return self._providers.get(name)

    def all(self) -> list[LLMProvider]:
        return list(self._providers.values())

    def configured(self) -> list[LLMProvider]:
        return [provider for provider in self._providers.values() if provider.configured]


@lru_cache
def get_provider_registry() -> ProviderRegistry:
    settings = get_settings()
    return ProviderRegistry(
        [
            GeminiProvider(settings.gemini_api_key),
            OpenAIProvider(settings.openai_api_key, settings.gateway_timeout_seconds),
        ]
    )
