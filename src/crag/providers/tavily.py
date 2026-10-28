"""Tavily-backed web-search provider."""

from datetime import datetime
from typing import Any

from tavily import AsyncTavilyClient

from crag.config import Settings
from crag.exceptions import ProviderError
from crag.models import WebSearchResult
from crag.providers._runtime import validate_timeout, with_timeout
from crag.providers.base import DEFAULT_PROVIDER_TIMEOUT_SECONDS, WebSearchProvider


class TavilySearchProvider(WebSearchProvider):
    """Search the web through Tavily and map results to CRAG models."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: AsyncTavilyClient | None = None,
    ) -> None:
        if settings.tavily_api_key is None:
            raise ValueError("Tavily API credentials are required")
        self._client = client or AsyncTavilyClient(
            api_key=settings.tavily_api_key.get_secret_value()
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: float = DEFAULT_PROVIDER_TIMEOUT_SECONDS,
    ) -> list[WebSearchResult]:
        if not query.strip():
            raise ValueError("query must not be blank")
        if max_results <= 0:
            raise ValueError("max_results must be greater than zero")
        validate_timeout(timeout)
        try:
            response = await with_timeout(
                self._client.search(query=query, max_results=max_results),
                timeout,
            )
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                raise ProviderError("Tavily returned malformed search results")
            return [self._map_result(item) for item in raw_results[:max_results]]
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("Tavily web search failed") from exc

    @staticmethod
    def _map_result(item: Any) -> WebSearchResult:
        if not isinstance(item, dict):
            raise ProviderError("Tavily returned a malformed search result")
        try:
            published_at = item.get("published_date")
            if isinstance(published_at, str):
                published_at = datetime.fromisoformat(
                    published_at.replace("Z", "+00:00")
                )
            return WebSearchResult(
                title=item["title"],
                url=item["url"],
                snippet=item["content"],
                published_at=published_at,
            )
        except Exception as exc:
            raise ProviderError("Tavily returned an invalid search result") from exc
