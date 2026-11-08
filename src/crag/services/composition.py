"""Explicit provider composition for configured CRAG runtime modes."""

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from crag.api import create_app
from crag.core.config import Settings, get_settings
from crag.core.exceptions import ConfigurationError, ProviderError
from crag.core.models import WebSearchResult
from crag.providers.base import (
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ChatProvider,
    EmbeddingProvider,
    WebSearchProvider,
)
from crag.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from crag.providers.openai import OpenAIChatProvider, OpenAIEmbeddingProvider
from crag.providers.qdrant import QdrantVectorStoreProvider
from crag.providers.tavily import TavilySearchProvider
from crag.rag.workflow import CragWorkflow


class DisabledWebSearchProvider(WebSearchProvider):
    """Make unavailable web fallback explicit in local-only deployments."""

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[WebSearchResult]:
        raise ProviderError("Web search is not configured")


async def build_workflow(
    settings: Settings | None = None,
    *,
    qdrant_client: AsyncQdrantClient | None = None,
    chat_provider: ChatProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    web_search_provider: WebSearchProvider | None = None,
) -> CragWorkflow:
    """Build providers and workflow without contacting services at import time."""
    active_settings = settings or get_settings()
    if not active_settings.qdrant_url:
        raise ConfigurationError(
            "CRAG_QDRANT_URL is required to build the configured workflow"
        )

    chat: ChatProvider | None = None
    embeddings: EmbeddingProvider | None = None
    vector_store: QdrantVectorStoreProvider | None = None
    uploaded_vector_store: QdrantVectorStoreProvider | None = None
    web_search: WebSearchProvider | None = None
    try:
        if active_settings.provider_mode == "local":
            chat = chat_provider or OllamaChatProvider(active_settings)
            embeddings = embedding_provider or OllamaEmbeddingProvider(active_settings)
            collection_name = active_settings.ollama_qdrant_collection_name
        else:
            chat = chat_provider or OpenAIChatProvider(active_settings)
            embeddings = embedding_provider or OpenAIEmbeddingProvider(active_settings)
            collection_name = active_settings.qdrant_collection_name

        vector = await embeddings.embed_query(
            "CRAG embedding dimension probe",
            timeout=DEFAULT_PROVIDER_TIMEOUT_SECONDS,
        )
        if web_search_provider is not None:
            web_search = web_search_provider
        elif active_settings.enable_tavily:
            web_search = TavilySearchProvider(active_settings)
        else:
            web_search = DisabledWebSearchProvider()

        vector_store = QdrantVectorStoreProvider(
            active_settings,
            embeddings,
            collection_name=collection_name,
            vector_size=len(vector),
            client=qdrant_client,
        )
        uploaded_vector_store = QdrantVectorStoreProvider(
            active_settings,
            embeddings,
            collection_name=active_settings.uploaded_collection_name,
            vector_size=len(vector),
            client=qdrant_client,
        )
        owned_resources = [
            resource
            for resource, injected in (
                (embeddings, embedding_provider),
                (chat, chat_provider),
                (web_search, web_search_provider),
            )
            if injected is None
        ]
        return CragWorkflow(
            vector_store=vector_store,
            chat=chat,
            web_search=web_search,
            owned_resources=owned_resources,
            web_search_enabled=active_settings.enable_tavily,
            additional_vector_stores=(uploaded_vector_store,),
        )
    except Exception:
        if vector_store is not None:
            await vector_store.close()
        if uploaded_vector_store is not None:
            await uploaded_vector_store.close()
        if embedding_provider is None and embeddings is not None:
            close = getattr(embeddings, "close", None)
            if close is not None:
                await close()
        if chat_provider is None and chat is not None:
            close = getattr(chat, "close", None)
            if close is not None:
                await close()
        if web_search_provider is None and web_search is not None:
            close = getattr(web_search, "close", None)
            if close is not None:
                await close()
        raise


async def build_app(
    settings: Settings | None = None,
    *,
    qdrant_client: AsyncQdrantClient | None = None,
    chat_provider: ChatProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    web_search_provider: WebSearchProvider | None = None,
) -> FastAPI:
    """Build the FastAPI application after explicit provider initialization."""
    workflow = await build_workflow(
        settings,
        qdrant_client=qdrant_client,
        chat_provider=chat_provider,
        embedding_provider=embedding_provider,
        web_search_provider=web_search_provider,
    )
    return create_app(workflow)
