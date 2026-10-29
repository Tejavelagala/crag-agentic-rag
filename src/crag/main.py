"""Uvicorn entry point for the configured CRAG application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from crag.api import WorkflowProtocol, create_app
from crag.composition import build_workflow
from crag.workflow import CragState


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


_lazy_workflow = _LazyWorkflow()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize providers at startup and close owned resources at shutdown."""
    _lazy_workflow.set_workflow(None)
    workflow = await build_workflow()
    try:
        _lazy_workflow.set_workflow(workflow)
        yield
    finally:
        _lazy_workflow.set_workflow(None)
        await workflow.close()


app = create_app(_lazy_workflow)
app.router.lifespan_context = lifespan
