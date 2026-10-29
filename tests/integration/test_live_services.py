"""Opt-in smoke tests for configured external CRAG services."""

import asyncio
import math
import os
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, models

from crag.api import create_app
from crag.config import Settings
from crag.ingestion import ingest_document
from crag.models import GradingDecision, WebSearchResult
from crag.providers.base import WebSearchProvider
from crag.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from crag.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from crag.providers.qdrant import QdrantVectorStoreProvider
from crag.providers.tavily import TavilySearchProvider
from crag.workflow import CragWorkflow

pytestmark = pytest.mark.integration

INTEGRATION_TIMEOUT_SECONDS = 20.0


def _require_opt_in() -> None:
    if os.getenv("CRAG_RUN_INTEGRATION_TESTS", "").lower() != "true":
        pytest.skip(
            "Live integration tests are disabled; set "
            "CRAG_RUN_INTEGRATION_TESTS=true to enable them."
        )


def _settings() -> Settings:
    _require_opt_in()
    try:
        return Settings(
            enable_openai=False,
            enable_tavily=False,
            enable_qdrant=False,
        )
    except ValidationError as exc:
        pytest.skip(f"CRAG integration configuration is invalid: {exc}")


def _require_secret(settings: Settings, name: str, value: object) -> None:
    if value is None:
        pytest.skip(f"{name} is not configured for live integration testing.")
    if hasattr(value, "get_secret_value"):
        value = value.get_secret_value()
    if not isinstance(value, str) or not value.strip():
        pytest.skip(f"{name} is not configured for live integration testing.")


def _require_qdrant_collection() -> str:
    collection = os.getenv("CRAG_INTEGRATION_QDRANT_COLLECTION", "").strip()
    if not collection:
        pytest.skip(
            "Set CRAG_INTEGRATION_QDRANT_COLLECTION to a dedicated Qdrant "
            "collection before running live integration tests."
        )
    if not collection.startswith("crag_integration_"):
        pytest.skip(
            "Qdrant integration requires a dedicated collection name beginning "
            "with 'crag_integration_'; no collection was modified."
        )
    return collection


def _require_ollama_qdrant_collection() -> str:
    collection = os.getenv(
        "CRAG_INTEGRATION_OLLAMA_QDRANT_COLLECTION",
        "crag_integration_ollama",
    ).strip()
    if not collection.startswith("crag_integration_"):
        pytest.skip(
            "Ollama Qdrant integration requires a dedicated collection name "
            "beginning with 'crag_integration_'; no collection was modified."
        )
    return collection


def _require_timeout() -> float:
    timeout_value = os.getenv(
        "CRAG_INTEGRATION_TIMEOUT_SECONDS",
        str(INTEGRATION_TIMEOUT_SECONDS),
    )
    try:
        timeout = float(timeout_value)
    except ValueError:
        pytest.skip("CRAG_INTEGRATION_TIMEOUT_SECONDS must be numeric.")
    if not math.isfinite(timeout) or timeout <= 0:
        pytest.skip(
            "CRAG_INTEGRATION_TIMEOUT_SECONDS must be finite and greater than zero."
        )
    return timeout


class FailingWebSearchProvider(WebSearchProvider):
    """Fail loudly if the workflow unexpectedly uses web fallback."""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: float = 30.0,
    ) -> list[WebSearchResult]:
        raise AssertionError("web search fallback was unexpectedly called")


def test_live_ollama_chat_and_embedding() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_OLLAMA_BASE_URL", settings.ollama_base_url)
    if not settings.ollama_chat_model.strip():
        pytest.skip("CRAG_OLLAMA_CHAT_MODEL is not configured.")
    if not settings.ollama_embedding_model.strip():
        pytest.skip("CRAG_OLLAMA_EMBEDDING_MODEL is not configured.")
    timeout = _require_timeout()

    async def exercise() -> None:
        chat = OllamaChatProvider(settings)
        embeddings = OllamaEmbeddingProvider(settings)

        answer = await chat.generate(
            "Reply with a short greeting.",
            timeout=timeout,
        )
        vector = await embeddings.embed_query(
            "CRAG local Ollama integration smoke test",
            timeout=timeout,
        )

        assert answer.strip()
        assert vector
        assert all(isinstance(value, float) for value in vector)

    asyncio.run(exercise())


def test_live_ollama_qdrant_upsert_and_retrieval() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_OLLAMA_BASE_URL", settings.ollama_base_url)
    _require_secret(settings, "CRAG_QDRANT_URL", settings.qdrant_url)
    collection = _require_ollama_qdrant_collection()
    timeout = _require_timeout()
    document_id = f"integration-ollama-{uuid4().hex}"
    chunk = ingest_document(
        "Ollama embeddings power local semantic retrieval for this CRAG test.",
        document_id=document_id,
        metadata={"test_run": document_id, "provider": "ollama"},
        parent_size=200,
        child_size=200,
        overlap=0,
    )[0]
    point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )

    async def exercise() -> None:
        embeddings = OllamaEmbeddingProvider(settings)
        vector = await embeddings.embed_query(chunk.text, timeout=timeout)
        provider = QdrantVectorStoreProvider(
            settings,
            embeddings,
            collection_name=collection,
            vector_size=len(vector),
            client=client,
        )
        try:
            await provider.upsert_chunks([chunk], timeout=timeout)
            evidence = await provider.retrieve(
                "How does local Ollama semantic retrieval work?",
                top_k=1,
                timeout=timeout,
            )
            assert evidence
            assert evidence[0].source_id == document_id
            assert evidence[0].text == chunk.text
            assert evidence[0].metadata["chunk_id"] == chunk.chunk_id
            assert evidence[0].metadata["test_run"] == document_id
            assert evidence[0].metadata["provider"] == "ollama"
        finally:
            await asyncio.wait_for(
                client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=[point_id]),
                    wait=True,
                ),
                timeout=timeout,
            )
            await client.close()

    asyncio.run(exercise())


def test_live_ollama_qdrant_workflow_and_api() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_OLLAMA_BASE_URL", settings.ollama_base_url)
    _require_secret(settings, "CRAG_QDRANT_URL", settings.qdrant_url)
    collection = _require_ollama_qdrant_collection()
    timeout = _require_timeout()
    document_id = f"integration-ollama-workflow-{uuid4().hex}"
    run_marker = uuid4().hex[:8]
    chunk = ingest_document(
        "Configuration fact: the chat model is qwen2.5:3b. "
        "Configuration fact: the embedding model is nomic-embed-text. "
        f"Integration verification marker: {run_marker}.",
        document_id=document_id,
        metadata={"test_run": document_id, "provider": "ollama"},
        parent_size=200,
        child_size=200,
        overlap=0,
    )[0]
    point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))
    question = (
        "Which chat and embedding models are configured for the local CRAG system? "
        f"Use the evidence with integration verification marker {run_marker}."
    )
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )

    async def exercise() -> None:
        embeddings = OllamaEmbeddingProvider(settings)
        vector = await embeddings.embed_query(chunk.text, timeout=timeout)
        vector_store = QdrantVectorStoreProvider(
            settings,
            embeddings,
            collection_name=collection,
            vector_size=len(vector),
            client=client,
        )
        workflow = CragWorkflow(
            vector_store=vector_store,
            chat=OllamaChatProvider(settings),
            web_search=FailingWebSearchProvider(),
            top_k=1,
            max_web_results=1,
            timeout=timeout,
        )
        try:
            await vector_store.upsert_chunks([chunk], timeout=timeout)
            state = await workflow.invoke(question)
            assert state["retrieved_evidence"]
            assert state["retrieved_evidence"][0].source_id == document_id
            assert state["final_answer"]
            assert state.get("error") is None
            assert state["fallback_used"] is False

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app(workflow)),
                base_url="http://test",
            ) as http_client:
                response = await http_client.post(
                    "/api/v1/ask",
                    json={"question": question},
                )
            assert response.status_code == 200
            body = response.json()
            assert body["answer"]
            assert body["fallback_used"] is False
            assert body["sources"]
            assert body["sources"][0]["kind"] == "internal"
            assert body["sources"][0]["source_id"] == document_id
            assert body["diagnostics"] == []
        finally:
            await asyncio.wait_for(
                client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=[point_id]),
                    wait=True,
                ),
                timeout=timeout,
            )
            await client.close()

    asyncio.run(exercise())


def test_live_openai_chat_structured_and_embedding() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_OPENAI_API_KEY", settings.openai_api_key)
    timeout = _require_timeout()

    async def exercise() -> None:
        chat = OpenAIChatProvider(settings)
        embeddings = OpenAIEmbeddingProvider(settings)

        answer = await chat.generate(
            "Reply with a short greeting.",
            timeout=timeout,
        )
        decision = await chat.generate_structured(
            "Is the following evidence relevant to the question? "
            "Question: What color is the sky? Evidence: The sky is blue.",
            output_type=GradingDecision,
            timeout=timeout,
        )
        vector = await embeddings.embed_query("CRAG integration smoke test")

        assert answer.strip()
        assert decision.decision in {"yes", "no"}
        assert decision.rationale.strip()
        assert vector
        assert all(isinstance(value, float) for value in vector)

    asyncio.run(exercise())


def test_live_qdrant_upsert_and_retrieval() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_QDRANT_URL", settings.qdrant_url)
    collection = _require_qdrant_collection()
    timeout = _require_timeout()
    document_id = f"integration-{uuid4().hex}"
    chunk = ingest_document(
        "CRAG live integration smoke-test evidence.",
        document_id=document_id,
        metadata={"test_run": document_id},
        parent_size=100,
        child_size=100,
        overlap=0,
    )[0]
    point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )

    async def exercise() -> None:
        embeddings = OpenAIEmbeddingProvider(settings)
        vector = await embeddings.embed_query(chunk.text, timeout=timeout)
        provider = QdrantVectorStoreProvider(
            settings,
            embeddings,
            collection_name=collection,
            vector_size=len(vector),
            client=client,
        )
        try:
            await provider.upsert_chunks([chunk], timeout=timeout)
            evidence = await provider.retrieve(
                "live integration smoke-test evidence",
                top_k=1,
                timeout=timeout,
            )
            assert evidence
            assert evidence[0].source_id == document_id
            assert evidence[0].metadata["chunk_id"] == chunk.chunk_id
            assert evidence[0].metadata["test_run"] == document_id
        finally:
            await asyncio.wait_for(
                client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=[point_id]),
                    wait=True,
                ),
                timeout=timeout,
            )
            await client.close()

    asyncio.run(exercise())


def test_live_tavily_search_maps_valid_results() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_TAVILY_API_KEY", settings.tavily_api_key)
    timeout = _require_timeout()

    async def exercise() -> None:
        provider = TavilySearchProvider(settings)
        results = await provider.search(
            "Python programming language",
            max_results=2,
            timeout=timeout,
        )

        assert 0 < len(results) <= 2
        for result in results:
            assert result.title.strip()
            assert result.snippet.strip()
            assert result.url.scheme in {"http", "https"}

    asyncio.run(exercise())


def test_live_workflow_and_api_with_real_providers() -> None:
    settings = _settings()
    _require_secret(settings, "CRAG_OPENAI_API_KEY", settings.openai_api_key)
    _require_secret(settings, "CRAG_QDRANT_URL", settings.qdrant_url)
    _require_secret(settings, "CRAG_TAVILY_API_KEY", settings.tavily_api_key)
    collection = _require_qdrant_collection()
    timeout = _require_timeout()
    document_id = f"integration-api-{uuid4().hex}"
    chunk = ingest_document(
        "The CRAG integration test marker identifies opt-in live tests.",
        document_id=document_id,
        metadata={"test_run": document_id},
        parent_size=200,
        child_size=200,
        overlap=0,
    )[0]
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )

    async def exercise() -> None:
        embeddings = OpenAIEmbeddingProvider(settings)
        vector = await embeddings.embed_query(chunk.text, timeout=timeout)
        vector_store = QdrantVectorStoreProvider(
            settings,
            embeddings,
            collection_name=collection,
            vector_size=len(vector),
            client=client,
        )
        workflow = CragWorkflow(
            vector_store=vector_store,
            chat=OpenAIChatProvider(settings),
            web_search=TavilySearchProvider(settings),
            top_k=1,
            max_web_results=1,
            timeout=timeout,
        )
        point_id = str(uuid5(NAMESPACE_URL, chunk.chunk_id))
        try:
            await vector_store.upsert_chunks([chunk], timeout=timeout)
            state = await workflow.invoke(
                "What does the CRAG integration test marker identify?"
            )
            assert state["final_answer"]
            assert state.get("error") is None

            response = await asyncio.to_thread(
                TestClient(create_app(workflow)).post,
                "/api/v1/ask",
                json={
                    "question": "What does the CRAG integration test marker identify?"
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["answer"]
            assert isinstance(body["fallback_used"], bool)
            assert isinstance(body["sources"], list)
        finally:
            await asyncio.wait_for(
                client.delete(
                    collection_name=collection,
                    points_selector=models.PointIdsList(points=[point_id]),
                    wait=True,
                ),
                timeout=timeout,
            )
            await client.close()

    asyncio.run(exercise())
