"""Deterministic tests for Ollama provider adapters."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from crag.config import Settings
from crag.exceptions import ProviderError, ProviderTimeoutError
from crag.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider


class FakeDecision(BaseModel):
    decision: str
    rationale: str


def ollama_settings() -> Settings:
    return Settings(
        _env_file=None,
        ollama_base_url="http://ollama.test",
        ollama_chat_model="chat-model",
        ollama_embedding_model="embedding-model",
    )


class FakeOllamaClient:
    async def chat(self, **kwargs: object) -> object:
        if "format" in kwargs:
            return {"message": {"content": '{"decision":"yes","rationale":"Relevant"}'}}
        return SimpleNamespace(message=SimpleNamespace(content="local response"))

    async def embed(self, **kwargs: object) -> object:
        inputs = kwargs.get("input", [])
        return {"embeddings": [[float(index), 1.0] for index, _ in enumerate(inputs)]}


def test_ollama_chat_and_structured_output() -> None:
    async def exercise() -> None:
        provider = OllamaChatProvider(ollama_settings(), client=FakeOllamaClient())

        assert await provider.generate("hello") == "local response"
        result = await provider.generate_structured("grade this", FakeDecision)
        assert result == FakeDecision(decision="yes", rationale="Relevant")

    asyncio.run(exercise())


def test_ollama_embeddings_preserve_count_and_dimension() -> None:
    async def exercise() -> None:
        provider = OllamaEmbeddingProvider(
            ollama_settings(),
            client=FakeOllamaClient(),
        )

        assert await provider.embed_documents(["one", "two"]) == [
            [0.0, 1.0],
            [1.0, 1.0],
        ]
        assert await provider.embed_query("query") == [0.0, 1.0]

    asyncio.run(exercise())


def test_ollama_rejects_malformed_outputs_and_translates_failures() -> None:
    class MalformedClient:
        async def chat(self, **_: object) -> object:
            return {"message": {"content": "not-json"}}

        async def embed(self, **_: object) -> object:
            return {"embeddings": [[1.0], ["invalid"]]}

    class FailingClient:
        async def chat(self, **_: object) -> object:
            raise RuntimeError("local provider failure")

        async def embed(self, **_: object) -> object:
            raise RuntimeError("local provider failure")

    async def exercise() -> None:
        malformed_chat = OllamaChatProvider(
            ollama_settings(),
            client=MalformedClient(),
        )
        with pytest.raises(ProviderError):
            await malformed_chat.generate_structured("grade", FakeDecision)

        malformed_embeddings = OllamaEmbeddingProvider(
            ollama_settings(),
            client=MalformedClient(),
        )
        with pytest.raises(ProviderError):
            await malformed_embeddings.embed_documents(["one", "two"])

        failing = OllamaChatProvider(ollama_settings(), client=FailingClient())
        with pytest.raises(ProviderError) as error:
            await failing.generate("hello")
        assert error.value.__cause__ is not None

    asyncio.run(exercise())


def test_ollama_translates_timeouts() -> None:
    class SlowClient:
        async def chat(self, **_: object) -> object:
            await asyncio.sleep(0.05)
            return {"message": {"content": "late"}}

    async def exercise() -> None:
        provider = OllamaChatProvider(ollama_settings(), client=SlowClient())
        with pytest.raises(ProviderTimeoutError) as error:
            await provider.generate("hello", timeout=0.001)
        assert error.value.__cause__ is not None

    asyncio.run(exercise())


def test_ollama_rejects_blank_inputs_and_models() -> None:
    settings = ollama_settings()

    with pytest.raises(ValueError, match="model"):
        OllamaChatProvider(settings, model=" ")
    with pytest.raises(ValueError, match="model"):
        OllamaEmbeddingProvider(settings, model=" ")

    async def exercise() -> None:
        chat = OllamaChatProvider(settings, client=FakeOllamaClient())
        embeddings = OllamaEmbeddingProvider(settings, client=FakeOllamaClient())
        with pytest.raises(ValueError, match="prompt"):
            await chat.generate(" ")
        with pytest.raises(ValueError, match="texts"):
            await embeddings.embed_documents([" "])

    asyncio.run(exercise())
