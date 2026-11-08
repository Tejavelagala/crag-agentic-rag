"""Ollama-backed chat and embedding providers."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from ollama import AsyncClient
from pydantic import BaseModel

from crag.core.config import Settings
from crag.core.exceptions import ProviderError
from crag.providers._runtime import validate_timeout, with_timeout
from crag.providers.base import (
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ChatProvider,
    EmbeddingProvider,
)

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class OllamaChatProvider(ChatProvider):
    """Generate text and structured output through a local Ollama model."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        client: AsyncClient | None = None,
    ) -> None:
        base_url = settings.ollama_base_url
        selected_model = model or settings.ollama_chat_model
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama base URL must not be blank")
        if not selected_model.strip():
            raise ValueError("Ollama chat model must not be blank")
        self._owns_client = client is None
        self._closed = False
        self._client = client or AsyncClient(host=base_url)
        self._model = selected_model

    async def close(self) -> None:
        """Close the underlying Ollama client."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.close()

    async def generate(
        self,
        prompt: str,
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        validate_timeout(timeout)
        try:
            response = await with_timeout(
                self._client.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0},
                ),
                timeout,
            )
            content = _response_value(_response_value(response, "message"), "content")
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("Ollama returned an empty chat response")
            return content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Ollama chat generation failed") from exc

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[OutputModelT],
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> OutputModelT:
        if not prompt.strip():
            raise ValueError("prompt must not be blank")
        validate_timeout(timeout)
        try:
            response = await with_timeout(
                self._client.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    format=output_type.model_json_schema(),
                    options={"temperature": 0},
                ),
                timeout,
            )
            content = _response_value(_response_value(response, "message"), "content")
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("Ollama returned empty structured output")
            return output_type.model_validate(json.loads(content))
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Ollama structured generation failed") from exc


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate query and document embeddings through a local Ollama model."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str | None = None,
        client: AsyncClient | None = None,
    ) -> None:
        base_url = settings.ollama_base_url
        selected_model = model or settings.ollama_embedding_model
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("Ollama base URL must not be blank")
        if not selected_model.strip():
            raise ValueError("Ollama embedding model must not be blank")
        self._owns_client = client is None
        self._closed = False
        self._client = client or AsyncClient(host=base_url)
        self._model = selected_model
        self._dimension: int | None = None

    async def close(self) -> None:
        """Close the underlying Ollama client."""
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.close()

    async def embed_query(
        self,
        text: str,
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[float]:
        embeddings = await self.embed_documents([text], timeout=timeout)
        return embeddings[0]

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("texts must not contain blank values")
        validate_timeout(timeout)
        try:
            response = await with_timeout(
                self._client.embed(model=self._model, input=list(texts)),
                timeout,
            )
            raw_embeddings = _response_value(response, "embeddings")
            if raw_embeddings is None:
                raw_embedding = _response_value(response, "embedding")
                raw_embeddings = [raw_embedding] if raw_embedding is not None else None
            if not isinstance(raw_embeddings, list) or len(raw_embeddings) != len(
                texts
            ):
                raise ProviderError("Ollama returned an unexpected embedding count")
            vectors = [self._validate_vector(vector) for vector in raw_embeddings]
            return vectors
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Ollama embedding generation failed") from exc

    def _validate_vector(self, vector: object) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise ProviderError("Ollama returned an invalid embedding")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ProviderError("Ollama returned a non-numeric embedding")
        values = [float(value) for value in vector]
        if self._dimension is None:
            self._dimension = len(values)
        elif len(values) != self._dimension:
            raise ProviderError("Ollama returned inconsistent embedding dimensions")
        return values


def _response_value(response: object, key: str) -> Any:
    if isinstance(response, Mapping):
        return response.get(key)
    return getattr(response, key, None)
