"""OpenAI-backed chat and embedding providers."""

from collections.abc import Sequence
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from crag.config import Settings
from crag.exceptions import ProviderError
from crag.providers._runtime import validate_timeout, with_timeout
from crag.providers.base import (
    DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ChatProvider,
    EmbeddingProvider,
)

OutputModelT = TypeVar("OutputModelT", bound=BaseModel)


class OpenAIChatProvider(ChatProvider):
    """Generate text and structured output through OpenAI chat completions."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str = "gpt-4o-mini",
        client: AsyncOpenAI | None = None,
    ) -> None:
        if settings.openai_api_key is None:
            raise ValueError("OpenAI API credentials are required")
        if not model.strip():
            raise ValueError("model must not be blank")
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )
        self._model = model

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
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                ),
                timeout,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("OpenAI returned an empty chat response")
            return content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI chat generation failed") from exc

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
                self._client.beta.chat.completions.parse(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=output_type,
                ),
                timeout,
            )
            parsed = response.choices[0].message.parsed
            if parsed is None:
                raise ProviderError("OpenAI returned no structured output")
            return (
                parsed
                if isinstance(parsed, output_type)
                else output_type.model_validate(parsed)
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI structured generation failed") from exc


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generate query and document embeddings through OpenAI."""

    def __init__(
        self,
        settings: Settings,
        *,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if settings.openai_api_key is None:
            raise ValueError("OpenAI API credentials are required")
        if not model.strip():
            raise ValueError("model must not be blank")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value()
        )
        self._model = model
        self._dimensions = dimensions
        self._dimension: int | None = dimensions

    async def embed_query(
        self,
        text: str,
        *,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[float]:
        embeddings = await self._embed([text], timeout=timeout)
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
        return await self._embed(texts, timeout=timeout)

    async def _embed(
        self,
        texts: Sequence[str],
        *,
        timeout: float,
    ) -> list[list[float]]:
        try:
            kwargs: dict[str, object] = {
                "model": self._model,
                "input": list(texts),
            }
            if self._dimensions is not None:
                kwargs["dimensions"] = self._dimensions
            response = await with_timeout(
                self._client.embeddings.create(**kwargs),
                timeout,
            )
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(texts):
                raise ProviderError("OpenAI returned an unexpected embedding count")
            vectors = [self._validate_vector(item.embedding) for item in data]
            return vectors
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("OpenAI embedding generation failed") from exc

    def _validate_vector(self, vector: object) -> list[float]:
        if not isinstance(vector, list) or not vector:
            raise ProviderError("OpenAI returned an invalid embedding")
        if not all(isinstance(value, (int, float)) for value in vector):
            raise ProviderError("OpenAI returned a non-numeric embedding")
        values = [float(value) for value in vector]
        if self._dimension is None:
            self._dimension = len(values)
        elif len(values) != self._dimension:
            raise ProviderError("OpenAI returned inconsistent embedding dimensions")
        return values
