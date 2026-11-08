"""Backward-compatible document extraction imports."""

from crag.ingestion.extraction import (
    DocumentExtractionError,
    ExtractedDocument,
    extract_document,
)

__all__ = ["DocumentExtractionError", "ExtractedDocument", "extract_document"]
