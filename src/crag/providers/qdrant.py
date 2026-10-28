"""Qdrant-backed vector-store provider."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient, models

from crag.config import Settings
from crag.exceptions import ProviderError
from crag.models import DocumentChunk, RetrievedEvidence
from crag.providers._runtime import validate_timeout, with_timeout
from crag.providers.base import (
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    EmbeddingProvider,
    VectorStoreProvider,
)


class QdrantVectorStoreProvider(VectorStoreProvider):
    """Persist and retrieve document chunks in a Qdrant collection."""

    def __init__(
        self,
        settings: Settings,
        embeddings: EmbeddingProvider,
        *,
        collection_name: str = "crag_chunks",
        vector_size: int,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        if not settings.qdrant_url:
            raise ValueError("Qdrant URL is required")
        if not collection_name.strip():
            raise ValueError("collection_name must not be blank")
        if vector_size <= 0:
            raise ValueError("vector_size must be greater than zero")
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key is not None
                else None
            ),
        )
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._vector_size = vector_size

    async def upsert_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> None:
        if not chunks:
            return
        validate_timeout(timeout)
        try:
            await with_timeout(self._ensure_collection(), timeout)
            vectors = await self._embeddings.embed_documents(
                [chunk.text for chunk in chunks],
                timeout=timeout,
            )
            if len(vectors) != len(chunks):
                raise ProviderError("Embedding count does not match chunk count")
            points = []
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._validate_vector(vector)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "parent_id": chunk.parent_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                points.append(
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                        vector=vector,
                        payload=payload,
                    )
                )
            await with_timeout(
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                    wait=True,
                ),
                timeout,
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Qdrant upsert failed") from exc

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[RetrievedEvidence]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        validate_timeout(timeout)
        try:
            vector = await self._embeddings.embed_query(query, timeout=timeout)
            self._validate_vector(vector)
            response = await with_timeout(
                self._client.query_points(
                    collection_name=self._collection_name,
                    query=vector,
                    limit=top_k,
                    with_payload=True,
                ),
                timeout,
            )
            points = getattr(response, "points", None)
            if not isinstance(points, list):
                raise ProviderError("Qdrant returned malformed query results")
            return [self._map_point(point) for point in points[:top_k]]
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Qdrant retrieval failed") from exc

    async def _ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=models.VectorParams(
                    size=self._vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return
        info = await self._client.get_collection(self._collection_name)
        configured_size = getattr(
            getattr(info, "config", None),
            "params",
            None,
        )
        configured_size = getattr(configured_size, "vectors", None)
        configured_size = getattr(configured_size, "size", None)
        if configured_size is not None and configured_size != self._vector_size:
            raise ProviderError(
                "Qdrant collection vector dimension does not match configuration"
            )

    def _validate_vector(self, vector: object) -> None:
        if not isinstance(vector, list) or len(vector) != self._vector_size:
            raise ProviderError("Embedding dimension does not match Qdrant collection")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ProviderError("Embedding vector contains non-numeric values")

    @staticmethod
    def _map_point(point: object) -> RetrievedEvidence:
        payload = getattr(point, "payload", None)
        score = getattr(point, "score", None)
        if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
            raise ProviderError("Qdrant returned a malformed payload")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ProviderError("Qdrant returned malformed chunk metadata")
        evidence_metadata = dict(metadata)
        for key in ("chunk_id", "document_id", "parent_id"):
            if payload.get(key) is not None:
                evidence_metadata[key] = payload[key]
        source_id = payload.get("document_id")
        if not isinstance(source_id, str):
            raise ProviderError("Qdrant payload is missing document identity")
        return RetrievedEvidence(
            text=payload["text"],
            source_id=source_id,
            metadata=evidence_metadata,
            retrieval_score=score,
        )
