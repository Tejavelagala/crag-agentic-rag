"""Document storage, extraction, indexing, and lifecycle orchestration."""

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Protocol

from crag.core.models import DocumentChunk
from crag.documents import DocumentRecord
from crag.ingestion import ingest_document
from crag.ingestion.extraction import extract_document
from crag.storage.registry import DocumentRegistry


class DocumentVectorStore(Protocol):
    async def upsert_chunks(
        self, chunks: list[DocumentChunk], *, timeout: float = 30.0
    ) -> None: ...

    async def delete_document(
        self, document_id: str, *, timeout: float = 30.0
    ) -> None: ...


class DocumentService:
    """Manage uploaded documents without embedding answering logic."""

    def __init__(
        self,
        *,
        registry: DocumentRegistry,
        vector_store: DocumentVectorStore,
        storage_dir: Path,
        max_upload_size: int,
        parent_size: int = 2_000,
        child_size: int = 500,
        overlap: int = 50,
    ) -> None:
        self.registry = registry
        self.vector_store = vector_store
        self.storage_dir = storage_dir
        self.max_upload_size = max_upload_size
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def upload(self, filename: str, content: bytes) -> DocumentRecord:
        if len(content) > self.max_upload_size:
            raise ValueError("The uploaded file exceeds the configured size limit.")
        safe_name = _safe_filename(filename)
        digest = hashlib.sha256(content).hexdigest()
        duplicate = await asyncio.to_thread(self.registry.duplicate, digest)
        if duplicate:
            raise FileExistsError(duplicate.document_id)
        extracted = await asyncio.to_thread(extract_document, safe_name, content)
        record = await asyncio.to_thread(
            self.registry.create,
            filename=safe_name,
            file_format=extracted.file_format,
            file_size=len(content),
            content_hash=digest,
        )
        path = self.storage_dir / f"{record.document_id}.{extracted.file_format}"
        try:
            await asyncio.to_thread(path.write_bytes, content)
            await asyncio.to_thread(
                self.registry.update,
                record.document_id,
                status="processing",
                extraction_metadata=extracted.metadata,
            )
            chunks = ingest_document(
                extracted.text,
                document_id=record.document_id,
                metadata={
                    "document_id": record.document_id,
                    "filename": safe_name,
                    "file_format": extracted.file_format,
                    "uploaded_document": True,
                },
                parent_size=self.parent_size,
                child_size=self.child_size,
                overlap=self.overlap,
            )
            await self.vector_store.upsert_chunks(chunks)
            await asyncio.to_thread(
                self.registry.save_chunks,
                record.document_id,
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "parent_id": chunk.parent_id,
                        "chunk_index": index,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                    for index, chunk in enumerate(chunks)
                ],
            )
            return await asyncio.to_thread(
                self.registry.update,
                record.document_id,
                status="indexed",
                chunk_count=len(chunks),
                indexing_metadata={"collection": record.collection_name},
                error=None,
            )
        except Exception as exc:
            cleanup_error: Exception | None = None
            try:
                await self.vector_store.delete_document(record.document_id)
            except Exception as cleanup_exc:
                cleanup_error = cleanup_exc
            error = _safe_error(exc)
            if cleanup_error is not None:
                error = f"{error}; vector cleanup failed: {_safe_error(cleanup_error)}"
            await asyncio.to_thread(
                self.registry.update,
                record.document_id,
                status="failed",
                error=error,
            )
            path.unlink(missing_ok=True)
            if cleanup_error is not None:
                raise ValueError(
                    "Document processing failed and indexed data cleanup failed."
                ) from cleanup_error
            raise ValueError("Document processing failed.") from exc

    async def get(self, document_id: str) -> DocumentRecord | None:
        return await asyncio.to_thread(self.registry.get, document_id)

    async def reindex(self, document_id: str) -> DocumentRecord:
        record = await self.get(document_id)
        if record is None:
            raise FileNotFoundError(document_id)
        path = self.storage_dir / f"{document_id}.{record.file_format}"
        try:
            content = await asyncio.to_thread(path.read_bytes)
            extracted = await asyncio.to_thread(
                extract_document, record.original_filename, content
            )
            chunks = ingest_document(
                extracted.text,
                document_id=document_id,
                metadata={
                    "document_id": document_id,
                    "filename": record.original_filename,
                    "file_format": extracted.file_format,
                    "uploaded_document": True,
                },
                parent_size=self.parent_size,
                child_size=self.child_size,
                overlap=self.overlap,
            )
            await self.vector_store.delete_document(document_id)
            await self.vector_store.upsert_chunks(chunks)
            await asyncio.to_thread(
                self.registry.save_chunks,
                document_id,
                [
                    {
                        "chunk_id": chunk.chunk_id,
                        "parent_id": chunk.parent_id,
                        "chunk_index": index,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                    for index, chunk in enumerate(chunks)
                ],
            )
            return await asyncio.to_thread(
                self.registry.update,
                document_id,
                status="indexed",
                chunk_count=len(chunks),
                extraction_metadata=extracted.metadata,
                error=None,
            )
        except FileNotFoundError:
            await asyncio.to_thread(
                self.registry.update,
                document_id,
                status="failed",
                error="Stored document file is missing.",
            )
            raise
        except Exception as exc:
            await asyncio.to_thread(
                self.registry.update,
                document_id,
                status="failed",
                error=_safe_error(exc),
            )
            raise ValueError("Document reindexing failed.") from exc

    async def list(
        self, *, offset: int, limit: int, search: str | None, status: str | None
    ):
        return await asyncio.to_thread(
            self.registry.list, offset=offset, limit=limit, search=search, status=status
        )

    async def chunks(self, document_id: str, *, offset: int, limit: int):
        return await asyncio.to_thread(self.registry.chunks, document_id, offset, limit)

    async def delete(self, document_id: str) -> bool:
        record = await self.get(document_id)
        if record is None:
            return False
        await asyncio.to_thread(self.registry.update, document_id, status="deleting")
        await self.vector_store.delete_document(document_id)
        path = self.storage_dir / f"{document_id}.{record.file_format}"
        path.unlink(missing_ok=True)
        await asyncio.to_thread(self.registry.delete, document_id)
        return True


def _safe_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name or name in {".", ".."}:
        raise ValueError("A valid filename is required.")
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name)[:255]


def _safe_error(exc: Exception) -> str:
    return type(exc).__name__ if not str(exc) else str(exc)[:240]
