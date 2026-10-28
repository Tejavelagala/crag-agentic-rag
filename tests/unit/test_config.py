"""Configuration behavior tests."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from crag.config import Settings


@pytest.fixture(autouse=True)
def clear_crag_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep configuration tests independent of the host environment."""
    for name in os.environ:
        if name.startswith("CRAG_"):
            monkeypatch.delenv(name)


def test_settings_load_defaults_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

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


def test_settings_load_valid_credentials_when_integration_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRAG_ENABLE_OPENAI", "true")
    monkeypatch.setenv("CRAG_OPENAI_API_KEY", "test-openai-key")

    settings = Settings()

    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "test-openai-key"


@pytest.mark.parametrize("value", [None, "", "   ", "\t\n"])
def test_enabled_provider_rejects_missing_or_blank_credentials(
    monkeypatch: pytest.MonkeyPatch,
    value: str | None,
) -> None:
    monkeypatch.setenv("CRAG_ENABLE_TAVILY", "true")
    if value is None:
        monkeypatch.delenv("CRAG_TAVILY_API_KEY", raising=False)
    else:
        monkeypatch.setenv("CRAG_TAVILY_API_KEY", value)

    with pytest.raises(ValidationError, match="CRAG_TAVILY_API_KEY") as error:
        Settings()
    assert "test-secret" not in str(error.value)


def test_disabled_provider_does_not_require_blank_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CRAG_ENABLE_OPENAI", "false")
    monkeypatch.setenv("CRAG_OPENAI_API_KEY", "   ")

    settings = Settings()

    assert settings.enable_openai is False


def test_settings_load_dotenv_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "CRAG_ENVIRONMENT=testing\n"
        "CRAG_DEBUG=true\n"
        "CRAG_ENABLE_OPENAI=true\n"
        "CRAG_OPENAI_API_KEY=dotenv-test-key\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.environment == "testing"
    assert settings.debug is True
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "dotenv-test-key"


def test_invalid_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_ENVIRONMENT", "invalid")

    with pytest.raises(ValidationError, match="environment"):
        Settings()


def test_qdrant_requires_non_blank_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRAG_ENABLE_QDRANT", "true")
    monkeypatch.setenv("CRAG_QDRANT_URL", "  ")

    with pytest.raises(ValidationError, match="CRAG_QDRANT_URL"):
        Settings()
