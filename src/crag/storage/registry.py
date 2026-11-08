"""Small transactional SQLite registry for uploaded documents."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from crag.documents import DocumentRecord, DocumentStatus


class DocumentRegistry:
    """Persist document metadata and chunk inspection records in SQLite."""

    def __init__(self, path: Path, collection_name: str) -> None:
        self._path = path
        self._collection_name = collection_name
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    normalized_filename TEXT NOT NULL,
                    file_format TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    extraction_metadata TEXT NOT NULL DEFAULT '{}',
                    indexing_metadata TEXT NOT NULL DEFAULT '{}',
                    error TEXT,
                    collection_name TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_hash
                    ON documents(content_hash);
                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    parent_id TEXT,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                """
            )

    def create(
        self,
        *,
        filename: str,
        file_format: str,
        file_size: int,
        content_hash: str,
    ) -> DocumentRecord:
        document_id = str(uuid4())
        now = datetime.now(UTC)
        record = DocumentRecord(
            document_id=document_id,
            original_filename=filename,
            normalized_filename=filename.strip().lower(),
            file_format=file_format,  # type: ignore[arg-type]
            file_size=file_size,
            content_hash=content_hash,
            uploaded_at=now,
            status="uploaded",
            collection_name=self._collection_name,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.document_id,
                    record.original_filename,
                    record.normalized_filename,
                    record.file_format,
                    record.file_size,
                    record.content_hash,
                    record.uploaded_at.isoformat(),
                    record.status,
                    record.chunk_count,
                    "{}",
                    "{}",
                    None,
                    record.collection_name,
                ),
            )
        return record

    def duplicate(self, content_hash: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def update(
        self,
        document_id: str,
        *,
        status: DocumentStatus | None = None,
        chunk_count: int | None = None,
        extraction_metadata: dict[str, Any] | None = None,
        indexing_metadata: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> DocumentRecord:
        record = self.get(document_id)
        if record is None:
            raise KeyError(document_id)
        updated = record.model_copy(
            update={
                "status": status or record.status,
                "chunk_count": chunk_count
                if chunk_count is not None
                else record.chunk_count,
                "extraction_metadata": extraction_metadata
                or record.extraction_metadata,
                "indexing_metadata": indexing_metadata or record.indexing_metadata,
                "error": error,
            }
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents SET status=?, chunk_count=?, extraction_metadata=?,
                indexing_metadata=?, error=? WHERE document_id=?
                """,
                (
                    updated.status,
                    updated.chunk_count,
                    _json(updated.extraction_metadata),
                    _json(updated.indexing_metadata),
                    updated.error,
                    document_id,
                ),
            )
        return updated

    def save_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
            )
            connection.executemany(
                "INSERT INTO document_chunks VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk["chunk_id"],
                        document_id,
                        chunk.get("parent_id"),
                        chunk["chunk_index"],
                        chunk["text"],
                        _json(chunk.get("metadata", {})),
                    )
                    for chunk in chunks
                ],
            )

    def chunks(
        self, document_id: str, offset: int, limit: int
    ) -> tuple[list[dict[str, Any]], int]:
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            rows = connection.execute(
                """
                SELECT chunk_id, parent_id, chunk_index, text, metadata
                FROM document_chunks WHERE document_id = ?
                ORDER BY chunk_index, chunk_id LIMIT ? OFFSET ?
                """,
                (document_id, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows], int(total)

    def delete(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM document_chunks WHERE document_id = ?", (document_id,)
            )
            connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )

    def list(
        self,
        *,
        offset: int,
        limit: int,
        search: str | None = None,
        status: str | None = None,
    ) -> tuple[list[DocumentRecord], int]:
        clauses: list[str] = []
        values: list[Any] = []
        if search:
            clauses.append("normalized_filename LIKE ?")
            values.append(f"%{search.strip().lower()}%")
        if status:
            clauses.append("status = ?")
            values.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM documents{where}", values
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM documents{where}
                ORDER BY uploaded_at DESC, document_id LIMIT ? OFFSET ?
                """,
                [*values, limit, offset],
            ).fetchall()
        return [self._row_to_record(row) for row in rows], int(total)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> DocumentRecord:
        import json

        return DocumentRecord(
            document_id=row["document_id"],
            original_filename=row["original_filename"],
            normalized_filename=row["normalized_filename"],
            file_format=row["file_format"],
            file_size=row["file_size"],
            content_hash=row["content_hash"],
            uploaded_at=row["uploaded_at"],
            status=row["status"],
            chunk_count=row["chunk_count"],
            extraction_metadata=json.loads(row["extraction_metadata"]),
            indexing_metadata=json.loads(row["indexing_metadata"]),
            error=row["error"],
            collection_name=row["collection_name"],
        )


def _json(value: object) -> str:
    import json

    return json.dumps(value, separators=(",", ":"))
