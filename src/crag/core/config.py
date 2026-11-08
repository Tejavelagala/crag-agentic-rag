"""Typed application configuration."""

from functools import lru_cache
from pathlib import Path
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

    provider_mode: Literal["local", "hosted"] = "local"
    enable_openai: bool = False
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    enable_tavily: bool = False
    tavily_api_key: SecretStr | None = Field(default=None, repr=False)
    enable_qdrant: bool = False
    qdrant_url: str | None = None
    qdrant_api_key: SecretStr | None = Field(default=None, repr=False)
    qdrant_collection_name: str = "crag_chunks"
    ollama_qdrant_collection_name: str = "crag_chunks_ollama"
    enable_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen2.5:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    document_storage_dir: Path = Path("data/documents")
    document_registry_path: Path = Path("data/documents.sqlite3")
    uploaded_collection_name: str = "crag_uploaded_chunks"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    ingestion_concurrency: int = 2
    document_parent_size: int = 2_000
    document_child_size: int = 500
    document_chunk_overlap: int = 50

    @model_validator(mode="after")
    def validate_enabled_integrations(self) -> "Settings":
        """Require only the credentials needed by enabled integrations."""
        missing: list[str] = []
        if self.enable_openai and not self._has_value(self.openai_api_key):
            missing.append("CRAG_OPENAI_API_KEY")
        if self.enable_tavily and not self._has_value(self.tavily_api_key):
            missing.append("CRAG_TAVILY_API_KEY")
        if self.enable_qdrant and not self._has_value(self.qdrant_url):
            missing.append("CRAG_QDRANT_URL")
        if self.provider_mode == "hosted" and not self._has_value(self.openai_api_key):
            missing.append("CRAG_OPENAI_API_KEY")
        if self.enable_ollama or self.provider_mode == "local":
            if not self._has_value(self.ollama_base_url):
                missing.append("CRAG_OLLAMA_BASE_URL")
            if not self._has_value(self.ollama_chat_model):
                missing.append("CRAG_OLLAMA_CHAT_MODEL")
            if not self._has_value(self.ollama_embedding_model):
                missing.append("CRAG_OLLAMA_EMBEDDING_MODEL")
        if missing:
            fields = ", ".join(missing)
            raise ValueError(
                f"Enabled integration settings must be non-empty: {fields}"
            )
        return self

    @model_validator(mode="after")
    def validate_document_settings(self) -> "Settings":
        """Validate bounded document-ingestion settings."""
        if self.max_upload_size_bytes <= 0:
            raise ValueError("CRAG_MAX_UPLOAD_SIZE_BYTES must be greater than zero")
        if self.ingestion_concurrency <= 0:
            raise ValueError("CRAG_INGESTION_CONCURRENCY must be greater than zero")
        if not self.uploaded_collection_name.strip():
            raise ValueError("CRAG_UPLOADED_COLLECTION_NAME must not be blank")
        if self.document_child_size > self.document_parent_size:
            raise ValueError("CRAG_DOCUMENT_CHILD_SIZE must not exceed parent size")
        if self.document_chunk_overlap >= self.document_child_size:
            raise ValueError("CRAG_DOCUMENT_CHUNK_OVERLAP must be less than child size")
        return self

    @staticmethod
    def _has_value(value: SecretStr | str | None) -> bool:
        """Return whether a setting contains non-whitespace content."""
        if value is None:
            return False
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        return bool(raw_value.strip())


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache, primarily for tests and process reconfiguration."""
    get_settings.cache_clear()
