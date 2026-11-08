"""Dependency protocols and application-state providers used by API routers."""

from pathlib import Path
from typing import Protocol

from fastapi import Request

from crag.rag.workflow import CragState


class WorkflowProtocol(Protocol):
    """Minimal workflow interface required by the HTTP API."""

    async def invoke(self, question: str) -> CragState:
        """Execute the workflow for one question."""


class DocumentServiceProtocol(Protocol):
    """Document service operations exposed to HTTP routes."""

    async def upload(self, filename: str, content: bytes): ...

    async def get(self, document_id: str): ...

    async def list(self, **kwargs: object): ...

    async def chunks(self, document_id: str, **kwargs: object): ...

    async def reindex(self, document_id: str): ...

    async def delete(self, document_id: str) -> bool: ...


def get_workflow(request: Request) -> WorkflowProtocol:
    """Resolve the lifespan-owned workflow from application state."""
    return request.app.state.workflow


def get_document_service(request: Request) -> DocumentServiceProtocol | None:
    """Resolve the optional lifespan-owned document service."""
    return request.app.state.document_service


def get_evaluation_report_path(request: Request) -> Path:
    """Resolve the configured evaluation report path."""
    return request.app.state.evaluation_report_path


def get_benchmark_report_path(request: Request) -> Path:
    """Resolve the configured benchmark report path."""
    return request.app.state.benchmark_report_path


__all__ = ["WorkflowProtocol"]
