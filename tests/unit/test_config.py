"""Configuration behavior tests."""

import pytest
from pydantic import ValidationError

from crag.config import Settings


def test_settings_load_defaults_without_credentials() -> None:
    settings = Settings()

    assert settings.environment == "development"
    assert settings.enable_openai is False
    assert settings.openai_api_key is None


def test_settings_load_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_ENVIRONMENT", "testing")
    monkeypatch.setenv("CRAG_DEBUG", "true")
    monkeypatch.setenv("CRAG_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.environment == "testing"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"


def test_enabled_provider_requires_its_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRAG_ENABLE_TAVILY", "true")

    with pytest.raises(ValidationError, match="CRAG_TAVILY_API_KEY"):
        Settings()


def test_qdrant_requires_url_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRAG_ENABLE_QDRANT", "true")

    with pytest.raises(ValidationError, match="CRAG_QDRANT_URL"):
        Settings()
