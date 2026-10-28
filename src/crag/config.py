"""Typed application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="CRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "crag-agentic-rag"
    environment: Literal["development", "testing", "staging", "production"] = (
        "development"
    )
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_json: bool = False

    enable_openai: bool = False
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    enable_tavily: bool = False
    tavily_api_key: SecretStr | None = Field(default=None, repr=False)
    enable_qdrant: bool = False
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = Field(default=None, repr=False)
    enable_ollama: bool = False
    ollama_base_url: str | None = None

    @model_validator(mode="after")
    def validate_enabled_integrations(self) -> "Settings":
        """Require only the credentials needed by enabled integrations."""
        missing: list[str] = []
        if self.enable_openai and self.openai_api_key is None:
            missing.append("CRAG_OPENAI_API_KEY")
        if self.enable_tavily and self.tavily_api_key is None:
            missing.append("CRAG_TAVILY_API_KEY")
        if self.enable_qdrant and not self.qdrant_url:
            missing.append("CRAG_QDRANT_URL")
        if missing:
            fields = ", ".join(missing)
            raise ValueError(f"Enabled integration settings are missing: {fields}")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache, primarily for tests and process reconfiguration."""
    get_settings.cache_clear()
