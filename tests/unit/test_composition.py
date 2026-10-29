"""Tests for explicit provider composition without external services."""

import asyncio

import pytest

from crag.composition import DisabledWebSearchProvider, build_workflow
from crag.config import Settings
from crag.exceptions import ConfigurationError, ProviderError
from crag.models import GradingDecision


class FakeChat:
    async def generate(
        self,
        prompt: str,
        *,
        timeout: float = 30.0,
    ) -> str:
        return "answer"

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[GradingDecision],
        *,
        timeout: float = 30.0,
    ) -> GradingDecision:
        return output_type(decision="yes", rationale="Relevant")


class FakeEmbeddings:
    async def embed_query(
        self,
        text: str,
        *,
        timeout: float = 30.0,
    ) -> list[float]:
        return [1.0, 2.0, 3.0]

    async def embed_documents(
        self,
        texts: list[str],
        *,
        timeout: float = 30.0,
    ) -> list[list[float]]:
        return [[1.0, 2.0, 3.0] for _ in texts]


class FakeQdrantClient:
    async def collection_exists(self, collection_name: str) -> bool:
        return False


def test_local_composition_selects_ollama_collection_without_network() -> None:
    settings = Settings(
        _env_file=None,
        qdrant_url="http://qdrant.test",
        ollama_qdrant_collection_name="crag_integration_ollama",
    )

    async def exercise() -> None:
        workflow = await build_workflow(
            settings,
            qdrant_client=FakeQdrantClient(),
            chat_provider=FakeChat(),
            embedding_provider=FakeEmbeddings(),
        )
        assert workflow._vector_store._collection_name == "crag_integration_ollama"
        assert workflow._vector_store._vector_size == 3
        assert workflow._chat.__class__ is FakeChat

    asyncio.run(exercise())


def test_composition_requires_qdrant_url() -> None:
    settings = Settings(_env_file=None)

    async def exercise() -> None:
        with pytest.raises(ConfigurationError, match="CRAG_QDRANT_URL"):
            await build_workflow(
                settings,
                chat_provider=FakeChat(),
                embedding_provider=FakeEmbeddings(),
            )

    asyncio.run(exercise())


def test_disabled_web_search_fails_explicitly() -> None:
    async def exercise() -> None:
        with pytest.raises(ProviderError, match="not configured"):
            await DisabledWebSearchProvider().search("question")

    asyncio.run(exercise())
