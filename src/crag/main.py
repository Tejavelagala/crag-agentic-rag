"""Uvicorn entry point for the configured CRAG application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from crag.api import WorkflowProtocol, create_app
from crag.core.config import get_settings
from crag.providers.qdrant import QdrantVectorStoreProvider
from crag.rag.workflow import CragState
from crag.services.composition import build_workflow
from crag.services.documents import DocumentService
from crag.storage.registry import DocumentRegistry


class _LazyWorkflow:
    """Delegate requests to the workflow initialized by the app lifespan."""

    def __init__(self) -> None:
        self._workflow: WorkflowProtocol | None = None

    def set_workflow(self, workflow: WorkflowProtocol | None) -> None:
        self._workflow = workflow

    @property
    def is_ready(self) -> bool:
        return self._workflow is not None

    async def invoke(self, question: str) -> CragState:
        if self._workflow is None:
            raise RuntimeError("CRAG workflow is not initialized")
        return await self._workflow.invoke(question)


class _LazyDocumentService:
    """Delegate document operations to the lifespan-owned service."""

    def __init__(self) -> None:
        self._service: DocumentService | None = None

    def set_service(self, service: DocumentService | None) -> None:
        self._service = service

    def _required(self) -> DocumentService:
        if self._service is None:
            raise RuntimeError("Document service is not initialized")
        return self._service

    async def upload(self, filename: str, content: bytes):
        return await self._required().upload(filename, content)

    async def get(self, document_id: str):
        return await self._required().get(document_id)

    async def list(self, **kwargs: object):
        return await self._required().list(**kwargs)

    async def chunks(self, document_id: str, **kwargs: object):
        return await self._required().chunks(document_id, **kwargs)

    async def delete(self, document_id: str) -> bool:
        return await self._required().delete(document_id)

    async def reindex(self, document_id: str):
        return await self._required().reindex(document_id)


_lazy_workflow = _LazyWorkflow()
_lazy_documents = _LazyDocumentService()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize providers at startup and close owned resources at shutdown."""
    _lazy_workflow.set_workflow(None)
    workflow = await build_workflow()
    try:
        _lazy_workflow.set_workflow(workflow)
        vector_stores = getattr(workflow, "additional_vector_stores", ())
        if vector_stores and isinstance(vector_stores[0], QdrantVectorStoreProvider):
            vector_store = vector_stores[0]
            settings = get_settings()
            _lazy_documents.set_service(
                DocumentService(
                    registry=DocumentRegistry(
                        settings.document_registry_path,
                        settings.uploaded_collection_name,
                    ),
                    vector_store=vector_store,
                    storage_dir=settings.document_storage_dir,
                    max_upload_size=settings.max_upload_size_bytes,
                    parent_size=settings.document_parent_size,
                    child_size=settings.document_child_size,
                    overlap=settings.document_chunk_overlap,
                )
            )
        yield
    finally:
        _lazy_documents.set_service(None)
        _lazy_workflow.set_workflow(None)
        await workflow.close()


app = create_app(_lazy_workflow, _lazy_documents)
app.router.lifespan_context = lifespan
app.mount(
    "/frontend",
    StaticFiles(directory=Path(__file__).resolve().parents[2] / "frontend", html=True),
    name="frontend",
)
