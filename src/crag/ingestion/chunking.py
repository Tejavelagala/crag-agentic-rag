"""Deterministic character-based document chunking."""

from hashlib import sha256

from pydantic import JsonValue

from crag.models import DocumentChunk


def ingest_document(
    text: str,
    *,
    document_id: str,
    metadata: dict[str, JsonValue] | None = None,
    parent_size: int = 2_000,
    child_size: int = 500,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Create ordered parent and overlapping child chunks from document text.

    Chunk sizes and overlap are measured in characters. Parents are contiguous
    non-overlapping sections. Children are created independently within each
    parent and overlap by ``overlap`` characters, so child text is not intended
    to be concatenated without de-duplication. Metadata is copied to every
    chunk; callers are responsible for excluding secrets from it.
    """
    _validate_inputs(
        text=text,
        document_id=document_id,
        parent_size=parent_size,
        child_size=child_size,
        overlap=overlap,
    )
    source_metadata = dict(metadata or {})
    chunks: list[DocumentChunk] = []

    for parent_position, parent_start in enumerate(range(0, len(text), parent_size)):
        parent_text = text[parent_start : parent_start + parent_size]
        parent_id = _chunk_id(
            document_id=document_id,
            role="parent",
            position=parent_position,
            text=parent_text,
        )
        chunks.append(
            DocumentChunk(
                chunk_id=parent_id,
                document_id=document_id,
                text=parent_text,
                metadata=source_metadata.copy(),
            )
        )

        child_step = child_size - overlap
        child_position = 0
        child_start = 0
        while True:
            child_text = parent_text[child_start : child_start + child_size]
            child_id = _chunk_id(
                document_id=document_id,
                role=f"child:{parent_id}",
                position=child_position,
                text=child_text,
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=child_id,
                    document_id=document_id,
                    text=child_text,
                    metadata=source_metadata.copy(),
                    parent_id=parent_id,
                )
            )
            if child_start + child_size >= len(parent_text):
                break
            next_start = min(
                child_start + child_step,
                len(parent_text) - child_size,
            )
            if next_start <= child_start:
                break
            child_start = next_start
            child_position += 1

    return chunks


def _validate_inputs(
    *,
    text: str,
    document_id: str,
    parent_size: int,
    child_size: int,
    overlap: int,
) -> None:
    if not text.strip():
        raise ValueError("text must not be empty or whitespace-only")
    if not document_id.strip():
        raise ValueError("document_id must not be empty or whitespace-only")
    if parent_size <= 0:
        raise ValueError("parent_size must be greater than zero")
    if child_size <= 0:
        raise ValueError("child_size must be greater than zero")
    if child_size > parent_size:
        raise ValueError("child_size must not exceed parent_size")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= child_size:
        raise ValueError("overlap must be less than child_size")


def _chunk_id(
    *,
    document_id: str,
    role: str,
    position: int,
    text: str,
) -> str:
    identity = f"{document_id}\0{role}\0{position}\0{text}".encode()
    digest = sha256(identity).hexdigest()[:16]
    return f"{document_id}:{role}:{position}:{digest}"
