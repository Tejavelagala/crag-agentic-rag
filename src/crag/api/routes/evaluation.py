"""Read-only evaluation report endpoints."""

from pathlib import Path

from fastapi import APIRouter, Depends

from crag.api.dependencies import (
    get_benchmark_report_path,
    get_evaluation_report_path,
)
from crag.api.legacy import (
    BenchmarkReportResponse,
    EvaluationReportResponse,
    _read_report,
)

router = APIRouter(tags=["evaluation"])


@router.get("/api/v1/evaluation", response_model=EvaluationReportResponse)
async def evaluation_report(
    path: Path = Depends(get_evaluation_report_path),  # noqa: B008
) -> EvaluationReportResponse:
    """Return the validated evaluation report."""
    return _read_report(path, EvaluationReportResponse)


@router.get(
    "/api/v1/evaluation/benchmark",
    response_model=BenchmarkReportResponse,
)
async def benchmark_report(
    path: Path = Depends(get_benchmark_report_path),  # noqa: B008
) -> BenchmarkReportResponse:
    """Return the validated benchmark report."""
    return _read_report(path, BenchmarkReportResponse)
