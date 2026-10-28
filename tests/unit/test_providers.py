"""Provider contract behavior tests."""

import asyncio
import inspect
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from crag.models import DocumentChunk, RetrievedEvidence, WebSearchResult
from crag.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    VectorStoreProvider,
    WebSearchProvider,
)

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class Answer(BaseModel):
    answer: str


class FakeChatProvider(ChatProvider):
    async def generate(self, prompt: str, *, timeout: float = 30.0) -> str:
        return f"response: {prompt} ({timeout})"

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[OutputModelT],
        *,
        timeout: float = 30.0,
    ) -> OutputModelT:
        return output_type(answer=f"{prompt} ({timeout})")


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_query(self, text: str, *, timeout: float = 30.0) -> list[float]:
        return [float(len(text)), timeout]

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        timeout: float = 30.0,
    ) -> list[list[float]]:
        return [[float(len(text)), timeout] for text in texts]


class FakeVectorStoreProvider(VectorStoreProvider):
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        timeout: float = 30.0,
    ) -> None:
        self.chunks.extend(chunks)

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        timeout: float = 30.0,
    ) -> list[RetrievedEvidence]:
        evidence = RetrievedEvidence(text=query, source_id=self.chunks[0].document_id)
        return [evidence][:top_k]


class FakeWebSearchProvider(WebSearchProvider):
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: float = 30.0,
    ) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title=query,
                url="https://example.com",
                snippet=f"timeout={timeout}",
            )
        ][:max_results]


def test_provider_contract_methods_are_async() -> None:
    for provider_type in (
        ChatProvider,
        EmbeddingProvider,
        VectorStoreProvider,
        WebSearchProvider,
    ):
        methods = [
            member
            for _, member in inspect.getmembers(provider_type, inspect.isfunction)
            if getattr(member, "__isabstractmethod__", False)
        ]
        assert methods
        assert all(inspect.iscoroutinefunction(method) for method in methods)


def test_fake_providers_satisfy_async_contracts_without_network() -> None:
    async def exercise() -> None:
        chat = FakeChatProvider()
        embeddings = FakeEmbeddingProvider()
        vector_store = FakeVectorStoreProvider()
        web_search = FakeWebSearchProvider()
        chunk = DocumentChunk(chunk_id="chunk", document_id="doc", text="text")

        assert await chat.generate("hello", timeout=2.0) == "response: hello (2.0)"
        structured = await chat.generate_structured("hello", Answer, timeout=2.0)
        assert structured.answer == "hello (2.0)"
        assert await embeddings.embed_query("hello", timeout=2.0) == [5.0, 2.0]
        assert await embeddings.embed_documents(["a", "bb"], timeout=2.0) == [
            [1.0, 2.0],
            [2.0, 2.0],
        ]
        await vector_store.upsert_chunks([chunk], timeout=2.0)
        evidence = await vector_store.retrieve("query", top_k=1, timeout=2.0)
        assert evidence[0].source_id == "doc"
        results = await web_search.search("query", max_results=1, timeout=2.0)
        assert results[0].snippet == "timeout=2.0"

    asyncio.run(exercise())
