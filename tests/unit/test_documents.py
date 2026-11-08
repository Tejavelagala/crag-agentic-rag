"""Offline document extraction, registry, and lifecycle tests."""

import asyncio
from pathlib import Path

import pytest

from crag.document_extraction import DocumentExtractionError, extract_document
from crag.document_service import DocumentService
from crag.documents_registry import DocumentRegistry


class FakeDocumentStore:
    def __init__(self) -> None:
        self.chunks = []
        self.deleted: list[str] = []

    async def upsert_chunks(self, chunks, *, timeout=30.0) -> None:
        self.chunks.extend(chunks)

    async def delete_document(self, document_id: str, *, timeout=30.0) -> None:
        self.deleted.append(document_id)
        self.chunks = [
            chunk for chunk in self.chunks if chunk.document_id != document_id
        ]


class FailingCleanupStore(FakeDocumentStore):
    async def upsert_chunks(self, chunks, *, timeout=30.0) -> None:
        raise RuntimeError("index unavailable")

    async def delete_document(self, document_id: str, *, timeout=30.0) -> None:
        raise RuntimeError("cleanup unavailable")


def test_text_extraction_normalizes_content() -> None:
    result = extract_document("notes.txt", b" first   line \n\n second ")

    assert result.file_format == "txt"
    assert result.text == "first line\nsecond"


def test_unsupported_and_invalid_text_are_rejected() -> None:
    with pytest.raises(DocumentExtractionError, match="Unsupported"):
        extract_document("notes.bin", b"content")
    with pytest.raises(DocumentExtractionError, match="UTF-8"):
        extract_document("notes.txt", b"\xff")
    with pytest.raises(DocumentExtractionError, match="PDF"):
        extract_document("notes.pdf", b"%PDF-not-a-valid-document")
    with pytest.raises(DocumentExtractionError, match="Unsupported"):
        extract_document("notes.docx", b"PK\x03\x04invalid")


def test_registry_persists_duplicates_lists_and_chunks(tmp_path: Path) -> None:
    registry = DocumentRegistry(tmp_path / "documents.sqlite3", "uploaded")
    record = registry.create(
        filename="Notes.TXT",
        file_format="txt",
        file_size=4,
        content_hash="hash",
    )
    registry.save_chunks(
        record.document_id,
        [
            {
                "chunk_id": "chunk-1",
                "parent_id": None,
                "chunk_index": 0,
                "text": "text",
                "metadata": {"filename": "Notes.TXT"},
            }
        ],
    )

    assert registry.duplicate("hash").document_id == record.document_id
    chunks, total = registry.chunks(record.document_id, 0, 10)
    assert total == 1
    assert chunks[0]["text"] == "text"


def test_service_indexes_duplicates_and_deletes_safely(tmp_path: Path) -> None:
    store = FakeDocumentStore()
    service = DocumentService(
        registry=DocumentRegistry(tmp_path / "documents.sqlite3", "uploaded"),
        vector_store=store,
        storage_dir=tmp_path / "files",
        max_upload_size=1000,
    )

    record = asyncio.run(service.upload("notes.txt", b"Useful document text."))
    assert record.status == "indexed"
    assert record.chunk_count > 0
    assert asyncio.run(service.get(record.document_id)).status == "indexed"
    reindexed = asyncio.run(service.reindex(record.document_id))
    assert reindexed.status == "indexed"
    assert len(store.chunks) > 0

    with pytest.raises(FileExistsError):
        asyncio.run(service.upload("copy.txt", b"Useful document text."))

    assert asyncio.run(service.delete(record.document_id)) is True
    assert asyncio.run(service.get(record.document_id)) is None
    assert asyncio.run(service.delete(record.document_id)) is False


def test_service_surfaces_vector_cleanup_failure(tmp_path: Path) -> None:
    service = DocumentService(
        registry=DocumentRegistry(tmp_path / "documents.sqlite3", "uploaded"),
        vector_store=FailingCleanupStore(),
        storage_dir=tmp_path / "files",
        max_upload_size=1000,
    )

    with pytest.raises(ValueError, match="cleanup failed"):
        asyncio.run(service.upload("notes.txt", b"Useful document text."))

    records, total = service.registry.list(offset=0, limit=10)
    assert total == 1
    assert records[0].status == "failed"
    assert "cleanup failed" in (records[0].error or "")
