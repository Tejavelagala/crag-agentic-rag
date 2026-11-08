"""RAG question endpoint."""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from crag.api.dependencies import WorkflowProtocol, get_workflow
from crag.api.legacy import (
    AskFailureResponse,
    AskRequest,
    AskResponse,
    _build_evidence,
    _build_grading,
    _build_sources,
    _failure_payload,
    _safe_diagnostics,
    _stage_statuses,
    _web_search_status,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rag"])


@router.post(
    "/api/v1/ask",
    response_model=AskResponse | AskFailureResponse,
    status_code=status.HTTP_200_OK,
)
async def ask(
    request: AskRequest,
    workflow: WorkflowProtocol = Depends(get_workflow),  # noqa: B008
) -> AskResponse:
    """Execute the corrective retrieval workflow."""
    try:
        state = await workflow.invoke(request.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("CRAG workflow request failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The CRAG workflow is temporarily unavailable.",
        ) from exc
    if state.get("error"):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_failure_payload(
                "The CRAG workflow could not produce an answer.", state
            ),
        )
    return AskResponse(
        answer=state.get("final_answer"),
        sources=_build_sources(state),
        fallback_used=state.get("fallback_used", False),
        diagnostics=_safe_diagnostics(state),
        retrieved_evidence=_build_evidence(state),
        grading_decisions=_build_grading(state),
        retrieved_chunk_count=len(state.get("retrieved_evidence", [])),
        stage_timings_ms=state.get("stage_timings_ms", {}),
        stage_statuses=_stage_statuses(state),
        web_search=_web_search_status(state),
    )
