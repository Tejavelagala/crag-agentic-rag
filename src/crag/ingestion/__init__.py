"""Document extraction, chunking, and ingestion primitives."""

from crag.ingestion.chunking import ingest_document
from crag.ingestion.extraction import (
    DocumentExtractionError,
    ExtractedDocument,
    extract_document,
)

__all__ = [
    "DocumentExtractionError",
    "ExtractedDocument",
    "extract_document",
    "ingest_document",
]
