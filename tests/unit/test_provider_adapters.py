"""Deterministic tests for concrete provider adapters."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from crag.config import Settings
from crag.exceptions import ProviderError, ProviderTimeoutError
from crag.ingestion import ingest_document
from crag.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from crag.providers.qdrant import QdrantVectorStoreProvider
from crag.providers.tavily import TavilySearchProvider


class FakeAnswer(BaseModel):
    answer: str


def settings_with_credentials() -> Settings:
    return Settings(
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key",
        qdrant_url="http://qdrant.test",
    )


class FakeCompletions:
    def __init__(self) -> None:
        self.parsed = FakeAnswer(answer="structured")

    async def create(self, **_: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="response"))]
        )

    async def parse(self, **_: object) -> object:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))]
        )


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        self.embeddings = SimpleNamespace(create=self.create_embeddings)
        self.embedding_data = [
            SimpleNamespace(index=1, embedding=[2.0, 3.0]),
            SimpleNamespace(index=0, embedding=[0.0, 1.0]),
        ]

    async def create_embeddings(self, **_: object) -> object:
        input_values = _.get("input", [])
        data = sorted(self.embedding_data, key=lambda item: item.index)
        return SimpleNamespace(data=data[: len(input_values)])


def test_openai_chat_and_structured_output_mapping() -> None:
    async def exercise() -> None:
        provider = OpenAIChatProvider(
            settings_with_credentials(),
            client=FakeOpenAIClient(),
        )

        assert await provider.generate("hello") == "response"
        result = await provider.generate_structured("hello", FakeAnswer)
        assert result == FakeAnswer(answer="structured")

    asyncio.run(exercise())


def test_openai_embedding_order_and_dimension_validation() -> None:
    async def exercise() -> None:
        client = FakeOpenAIClient()
        provider = OpenAIEmbeddingProvider(
            settings_with_credentials(),
            client=client,
        )

        assert await provider.embed_documents(["first", "second"]) == [
            [0.0, 1.0],
            [2.0, 3.0],
        ]
        assert await provider.embed_query("query") == [0.0, 1.0]

        client.embedding_data = [
            SimpleNamespace(index=0, embedding=[1.0]),
            SimpleNamespace(index=1, embedding=[2.0, 3.0]),
        ]
        with pytest.raises(ProviderError, match="dimensions"):
            await provider.embed_documents(["first", "second"])

    asyncio.run(exercise())


def test_provider_timeout_and_failure_are_translated() -> None:
    class SlowClient:
        async def search(self, **_: object) -> object:
            await asyncio.sleep(0.05)
            return {"results": []}

    class FailingClient:
        async def search(self, **_: object) -> object:
            raise RuntimeError("network failure")

    async def exercise() -> None:
        timeout_provider = TavilySearchProvider(
            settings_with_credentials(),
            client=SlowClient(),
        )
        with pytest.raises(ProviderTimeoutError) as timeout_error:
            await timeout_provider.search("query", timeout=0.001)
        assert timeout_error.value.__cause__ is not None

        failing_provider = TavilySearchProvider(
            settings_with_credentials(),
            client=FailingClient(),
        )
        with pytest.raises(ProviderError) as provider_error:
            await failing_provider.search("query")
        assert provider_error.value.__cause__ is not None

    asyncio.run(exercise())


def test_tavily_maps_results_and_honors_limit() -> None:
    class FakeTavilyClient:
        async def search(self, **_: object) -> dict[str, object]:
            return {
                "results": [
                    {
                        "title": "First",
                        "url": "https://example.com/1",
                        "content": "One",
                        "published_date": "2026-01-01T00:00:00Z",
                    },
                    {
                        "title": "Second",
                        "url": "https://example.com/2",
                        "content": "Two",
                    },
                ]
            }

    async def exercise() -> None:
        provider = TavilySearchProvider(
            settings_with_credentials(),
            client=FakeTavilyClient(),
        )
        results = await provider.search("query", max_results=1)
        assert len(results) == 1
        assert results[0].title == "First"
        assert results[0].published_at is not None

        with pytest.raises(ValueError, match="max_results"):
            await provider.search("query", max_results=0)

    asyncio.run(exercise())


class FakeEmbeddingProvider:
    async def embed_documents(
        self,
        texts: list[str],
        *,
        timeout: float = 30.0,
    ) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    async def embed_query(
        self,
        text: str,
        *,
        timeout: float = 30.0,
    ) -> list[float]:
        return [float(len(text)), 1.0]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.points: list[object] = []
        self.created = False

    async def collection_exists(self, _: str) -> bool:
        return self.created

    async def create_collection(self, **_: object) -> None:
        self.created = True

    async def get_collection(self, _: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=SimpleNamespace(size=2))
            )
        )

    async def upsert(self, *, points: list[object], **_: object) -> None:
        self.points = points

    async def query_points(self, **_: object) -> object:
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    score=0.9,
                    payload=self.points[0].payload,
                )
            ]
        )


def test_qdrant_preserves_chunk_payload_and_parent_relationship() -> None:
    async def exercise() -> None:
        client = FakeQdrantClient()
        provider = QdrantVectorStoreProvider(
            settings_with_credentials(),
            FakeEmbeddingProvider(),
            vector_size=2,
            client=client,
        )
        chunks = ingest_document(
            "some document text",
            document_id="doc-1",
            metadata={"source": "test"},
            parent_size=100,
            child_size=10,
            overlap=2,
        )

        await provider.upsert_chunks(chunks)
        client.points = [client.points[1]]
        evidence = await provider.retrieve("document", top_k=1)

        assert evidence[0].text == chunks[1].text
        assert evidence[0].source_id == "doc-1"
        assert evidence[0].metadata["source"] == "test"
        assert evidence[0].metadata["parent_id"] == chunks[1].parent_id

    asyncio.run(exercise())


def test_qdrant_rejects_embedding_dimension_mismatch() -> None:
    class WrongDimensions(FakeEmbeddingProvider):
        async def embed_documents(
            self,
            texts: list[str],
            *,
            timeout: float = 30.0,
        ) -> list[list[float]]:
            return [[1.0] for _ in texts]

    async def exercise() -> None:
        provider = QdrantVectorStoreProvider(
            settings_with_credentials(),
            WrongDimensions(),
            vector_size=2,
            client=FakeQdrantClient(),
        )
        chunks = ingest_document("text", document_id="doc-1")

        with pytest.raises(ProviderError, match="dimension"):
            await provider.upsert_chunks(chunks)

    asyncio.run(exercise())


def test_adapters_reject_missing_credentials_and_invalid_outputs() -> None:
    settings = Settings()
    with pytest.raises(ValueError, match="credentials"):
        OpenAIChatProvider(settings)
    with pytest.raises(ValueError, match="credentials"):
        TavilySearchProvider(settings)

    class MalformedClient:
        async def search(self, **_: object) -> dict[str, object]:
            return {"results": [{"title": "missing fields"}]}

    async def exercise() -> None:
        provider = TavilySearchProvider(
            settings_with_credentials(),
            client=MalformedClient(),
        )
        with pytest.raises(ProviderError):
            await provider.search("query")

    asyncio.run(exercise())
