"""Typed document registry models and ingestion states."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal["uploaded", "processing", "indexed", "failed", "deleting"]
DocumentFormat = Literal["pdf", "docx", "txt"]


class DocumentRecord(BaseModel):
    """Public document metadata stored by the registry."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    original_filename: str
    normalized_filename: str
    file_format: DocumentFormat
    file_size: int
    content_hash: str
    uploaded_at: datetime
    status: DocumentStatus
    chunk_count: int = 0
    extraction_metadata: dict[str, str | int] = Field(default_factory=dict)
    indexing_metadata: dict[str, str | int] = Field(default_factory=dict)
    error: str | None = None
    collection_name: str


class DocumentChunkRecord(BaseModel):
    """Safe chunk inspection response without embeddings."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    parent_id: str | None
    chunk_index: int
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)
