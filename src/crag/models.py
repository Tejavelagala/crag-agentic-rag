"""Validated data models shared by CRAG providers and workflows."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, JsonValue, field_validator


class DocumentChunk(BaseModel):
    """A source document fragment prepared for retrieval."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    parent_id: str | None = Field(default=None, min_length=1)

    @field_validator("chunk_id", "document_id", "text", "parent_id")
    @classmethod
    def reject_blank_text(cls, value: str | None) -> str | None:
        """Reject values that contain no meaningful content."""
        if value is not None and not value.strip():
            raise ValueError("value must not be blank")
        return value


class RetrievedEvidence(BaseModel):
    """Evidence returned by a vector or retrieval provider."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    retrieval_score: float | None = None

    @field_validator("text", "source_id")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject values that contain no meaningful content."""
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class WebSearchResult(BaseModel):
    """A single result returned by a web-search provider."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1)
    url: HttpUrl
    snippet: str = Field(min_length=1)
    published_at: datetime | None = None

    @field_validator("title", "snippet")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject values that contain no meaningful content."""
        if not value.strip():
            raise ValueError("value must not be blank")
        return value


class GradingDecision(BaseModel):
    """A structured binary decision and its supporting rationale."""

    model_config = ConfigDict(frozen=True)

    decision: Literal["yes", "no"]
    rationale: str = Field(min_length=1)

    @field_validator("rationale")
    @classmethod
    def reject_blank_rationale(cls, value: str) -> str:
        """Require a meaningful explanation for the decision."""
        if not value.strip():
            raise ValueError("rationale must not be blank")
        return value
