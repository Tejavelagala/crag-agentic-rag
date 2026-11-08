"""Health endpoints."""

from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from crag.api.dependencies import WorkflowProtocol, get_workflow
from crag.api.legacy import HealthResponse
from crag.services.health import get_health_status

router = APIRouter(tags=["health"])


@router.get("/health/live", response_model=HealthResponse)
async def live() -> HealthResponse:
    """Report application liveness checks."""
    health = get_health_status()
    return HealthResponse(
        status="healthy", service=health.service, checks=health.checks
    )


@router.get("/health/ready", response_model=HealthResponse)
async def ready(
    workflow: WorkflowProtocol = Depends(get_workflow),  # noqa: B008
) -> HealthResponse | JSONResponse:
    """Report workflow configuration readiness."""
    workflow_configured = getattr(workflow, "is_ready", True)
    if not workflow_configured:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "not_ready",
                "service": "crag",
                "checks": {"application": True, "workflow_configured": False},
            },
        )
    return HealthResponse(
        status="ready",
        checks={"application": True, "workflow_configured": True},
    )
