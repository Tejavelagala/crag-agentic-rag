"""Provider-neutral asynchronous contracts for CRAG integrations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from crag.models import DocumentChunk, RetrievedEvidence, WebSearchResult

DEFAULT_PROVIDER_TIMEOUT_SECONDS = 30.0
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class ChatProvider(ABC):
    """Generate text or validated structured output from a chat model.

    Implementations must enforce the supplied timeout for their network
    operation. The default is a contract-level value, not enforcement.
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> str:
        """Generate a normal text response."""

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        output_type: type[OutputModelT],
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> OutputModelT:
        """Generate and validate a response using a Pydantic model."""


class EmbeddingProvider(ABC):
    """Create vector embeddings for queries and batches of text."""

    @abstractmethod
    async def embed_query(
        self,
        text: str,
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[float]:
        """Embed one query string."""

    @abstractmethod
    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[list[float]]:
        """Embed a batch of document strings in input order."""


class VectorStoreProvider(ABC):
    """Store chunks and retrieve relevant evidence."""

    @abstractmethod
    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> None:
        """Persist document chunks."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[RetrievedEvidence]:
        """Retrieve relevant evidence for a query."""


class WebSearchProvider(ABC):
    """Search the web and return typed result models."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[WebSearchResult]:
        """Search for a query."""
