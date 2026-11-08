"""FastAPI application factory and backward-compatible API exports."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from crag.api.dependencies import WorkflowProtocol
from crag.api.legacy import (
    AskFailureResponse,
    AskRequest,
    AskResponse,
    BenchmarkReportResponse,
    BenchmarkResultResponse,
    BenchmarkVariantResponse,
    DocumentChunkPage,
    DocumentPage,
    DocumentUploadResult,
    EvaluationCaseResponse,
    EvaluationReportResponse,
    EvidenceReference,
    GradingReference,
    HealthResponse,
    SourceReference,
    WebSearchStatus,
)
from crag.api.middleware.error_handler import validation_error_handler
from crag.api.routes.documents import router as documents_router
from crag.api.routes.evaluation import router as evaluation_router
from crag.api.routes.health import router as health_router
from crag.api.routes.rag import router as rag_router

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_REPORT_PATH = PROJECT_ROOT / "evaluation" / "results" / "latest.json"
BENCHMARK_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "grader_benchmark.json"
)


def create_app(workflow: WorkflowProtocol, document_service=None):
    """Create the application by assembling independent API routers."""
    if workflow is None:
        raise ValueError("workflow is required")
    app = FastAPI(title="CRAG API", version="0.1.0")
    app.state.workflow = workflow
    app.state.document_service = document_service
    app.state.evaluation_report_path = EVALUATION_REPORT_PATH
    app.state.benchmark_report_path = BENCHMARK_REPORT_PATH
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(rag_router)
    app.include_router(documents_router)
    app.include_router(health_router)
    app.include_router(evaluation_router)
    return app


__all__ = [
    "AskFailureResponse",
    "AskRequest",
    "AskResponse",
    "BenchmarkReportResponse",
    "BenchmarkResultResponse",
    "BenchmarkVariantResponse",
    "DocumentChunkPage",
    "DocumentPage",
    "DocumentUploadResult",
    "EvaluationCaseResponse",
    "EvaluationReportResponse",
    "EvidenceReference",
    "GradingReference",
    "HealthResponse",
    "SourceReference",
    "WebSearchStatus",
    "WorkflowProtocol",
    "create_app",
]
