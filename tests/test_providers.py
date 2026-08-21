from types import SimpleNamespace

import pytest

from app.providers.base import (
    ProviderChatInput,
    ProviderEmbeddingInput,
    ProviderStructuredInput,
)
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider


class FakeGeminiInteractions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        structured = "response_format" in kwargs
        return SimpleNamespace(
            id="gemini-response-1",
            output_text='{"answer":"ok"}' if structured else "gemini reply",
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=4,
                total_tokens=16,
                cached_input_tokens=2,
            ),
        )


class FakeGeminiModels:
    def embed_content(self, **kwargs):
        assert kwargs["model"] == "gemini-embedding-001"
        return SimpleNamespace(
            id="gemini-embed-1",
            embeddings=[SimpleNamespace(values=[0.1, 0.2]), SimpleNamespace(values=[0.3, 0.4])],
            usage_metadata=SimpleNamespace(
                prompt_token_count=8,
                candidates_token_count=0,
                total_token_count=8,
                cached_content_token_count=0,
            ),
        )


class FakeGeminiClient:
    def __init__(self) -> None:
        self.interactions = FakeGeminiInteractions()
        self.models = FakeGeminiModels()


class FakeOpenAIResponses:
    async def create(self, **kwargs):
        structured = "text" in kwargs
        return SimpleNamespace(
            id="openai-response-1",
            output_text='{"answer":"ok"}' if structured else "openai reply",
            usage=SimpleNamespace(
                input_tokens=20,
                output_tokens=5,
                total_tokens=25,
                input_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
        )


class FakeOpenAIEmbeddings:
    async def create(self, **kwargs):
        assert kwargs["model"] == "text-embedding-3-small"
        return SimpleNamespace(
            id="openai-embed-1",
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])],
            usage=SimpleNamespace(prompt_tokens=9, total_tokens=9),
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeOpenAIResponses()
        self.embeddings = FakeOpenAIEmbeddings()


@pytest.mark.anyio
async def test_gemini_provider_normalizes_chat_structured_and_embeddings() -> None:
    provider = GeminiProvider("test-key")
    provider._client = FakeGeminiClient()

    chat = await provider.chat(
        ProviderChatInput(
            model="gemini-3.5-flash",
            messages=[
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hello"},
            ],
        )
    )
    assert chat.content == "gemini reply"
    assert chat.usage.total_tokens == 16
    assert chat.usage.cached_input_tokens == 2

    structured = await provider.structured(
        ProviderStructuredInput(
            model="gemini-3.5-flash",
            messages=[{"role": "user", "content": "Extract"}],
            schema_name="answer",
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        )
    )
    assert structured.structured_data == {"answer": "ok"}

    embedded = await provider.embed(
        ProviderEmbeddingInput(
            model="gemini-embedding-001",
            inputs=["hello", "world"],
        )
    )
    assert embedded.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert embedded.usage.total_tokens == 8


@pytest.mark.anyio
async def test_openai_provider_normalizes_chat_structured_and_embeddings() -> None:
    provider = OpenAIProvider("test-key", 10)
    provider._client = FakeOpenAIClient()

    chat = await provider.chat(
        ProviderChatInput(
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "Hello"}],
            max_output_tokens=100,
        )
    )
    assert chat.content == "openai reply"
    assert chat.usage.total_tokens == 25
    assert chat.usage.cached_input_tokens == 3

    structured = await provider.structured(
        ProviderStructuredInput(
            model="gpt-5.6-terra",
            messages=[{"role": "user", "content": "Extract"}],
            schema_name="answer",
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        )
    )
    assert structured.structured_data == {"answer": "ok"}

    embedded = await provider.embed(
        ProviderEmbeddingInput(
            model="text-embedding-3-small",
            inputs=["hello", "world"],
        )
    )
    assert embedded.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert embedded.usage.total_tokens == 9


def test_provider_configuration_and_error_classification() -> None:
    gemini = GeminiProvider("")
    openai = OpenAIProvider("", 10)

    with pytest.raises(Exception) as gemini_exc:
        gemini._get_client()
    assert getattr(gemini_exc.value, "code", None) == "GEMINI_API_KEY_MISSING"

    with pytest.raises(Exception) as openai_exc:
        openai._get_client()
    assert getattr(openai_exc.value, "code", None) == "OPENAI_API_KEY_MISSING"

    gemini_rate_limit = gemini._classify(
        SimpleNamespace(status_code=429, __str__=lambda self: "rate")
    )
    assert gemini_rate_limit.retryable is True

    openai_bad_request = openai._classify(
        SimpleNamespace(status_code=400, __str__=lambda self: "bad")
    )
    assert openai_bad_request.retryable is False
