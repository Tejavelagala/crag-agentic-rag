"""Document management endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from crag.api.dependencies import DocumentServiceProtocol, get_document_service
from crag.api.legacy import DocumentChunkPage, DocumentPage, DocumentUploadResult
from crag.documents import DocumentChunkRecord, DocumentRecord

router = APIRouter(tags=["documents"])


def _required(
    service: DocumentServiceProtocol | None,
) -> DocumentServiceProtocol:
    if service is None:
        raise HTTPException(
            status_code=503, detail="Document ingestion is not configured."
        )
    return service


@router.post("/api/v1/documents", response_model=list[DocumentUploadResult])
async def upload_documents(
    files: Annotated[list[UploadFile], File(...)],
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> list[DocumentUploadResult]:
    """Upload and index one or more documents."""
    document_service = _required(service)
    results: list[DocumentUploadResult] = []
    for upload in files:
        try:
            content = await upload.read()
            document = await document_service.upload(upload.filename or "", content)
            results.append(
                DocumentUploadResult(filename=upload.filename or "", document=document)
            )
        except FileExistsError as exc:
            results.append(
                DocumentUploadResult(
                    filename=upload.filename or "",
                    error="Duplicate document content.",
                    duplicate_document_id=str(exc),
                )
            )
        except (ValueError, OSError) as exc:
            results.append(
                DocumentUploadResult(filename=upload.filename or "", error=str(exc))
            )
        finally:
            await upload.close()
    return results


@router.get("/api/v1/documents", response_model=DocumentPage)
async def list_documents(
    offset: int = 0,
    limit: int = 20,
    search: str | None = None,
    document_status: str | None = None,
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> DocumentPage:
    """List documents with pagination and optional filters."""
    document_service = _required(service)
    if offset < 0 or limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="Invalid pagination.")
    items, total = await document_service.list(
        offset=offset, limit=limit, search=search, status=document_status
    )
    return DocumentPage(items=items, offset=offset, limit=limit, total=total)


@router.get("/api/v1/documents/search", response_model=DocumentPage)
async def search_documents(
    offset: int = 0,
    limit: int = 20,
    q: str | None = None,
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> DocumentPage:
    """Search documents by filename metadata."""
    return await list_documents(offset=offset, limit=limit, search=q, service=service)


@router.get(
    "/api/v1/documents/{document_id}/chunks",
    response_model=DocumentChunkPage,
)
async def get_document_chunks(
    document_id: str,
    offset: int = 0,
    limit: int = 20,
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> DocumentChunkPage:
    """Inspect persisted chunks for one document."""
    document_service = _required(service)
    if await document_service.get(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if offset < 0 or limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="Invalid pagination.")
    items, total = await document_service.chunks(
        document_id, offset=offset, limit=limit
    )
    return DocumentChunkPage(
        items=[DocumentChunkRecord.model_validate(item) for item in items],
        offset=offset,
        limit=limit,
        total=total,
    )


@router.post("/api/v1/documents/{document_id}/reindex", response_model=DocumentRecord)
async def reindex_document(
    document_id: str,
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> DocumentRecord:
    """Rebuild vectors for one stored document."""
    document_service = _required(service)
    try:
        return await document_service.reindex(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/api/v1/documents/{document_id}", response_model=DocumentRecord)
async def get_document(
    document_id: str,
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> DocumentRecord:
    """Return one document registry record."""
    document_service = _required(service)
    document = await document_service.get(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document


@router.delete("/api/v1/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    service: DocumentServiceProtocol | None = Depends(get_document_service),  # noqa: B008
) -> None:
    """Delete one document and its indexed vectors."""
    document_service = _required(service)
    if not await document_service.delete(document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
