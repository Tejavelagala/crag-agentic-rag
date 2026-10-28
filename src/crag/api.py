"""FastAPI application for the corrective retrieval workflow."""

import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from crag.health import HealthStatus, get_health_status
from crag.models import JsonValue
from crag.workflow import CragState, CragWorkflow

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """Request body for a corrective retrieval question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=10_000)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        """Reject whitespace-only questions."""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class SourceReference(BaseModel):
    """Safe source information returned with an answer."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["internal", "external"]
    source_id: str | None = None
    title: str | None = None
    url: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Safe HTTP response for a corrective retrieval request."""

    answer: str | None
    sources: list[SourceReference]
    fallback_used: bool
    diagnostics: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health response that reports only checks performed by the API."""

    status: Literal["healthy", "ready"]
    service: str = "crag"
    checks: dict[str, bool]


def create_app(workflow: CragWorkflow) -> FastAPI:
    """Create an API application using an explicitly injected workflow."""
    if workflow is None:
        raise ValueError("workflow is required")

    app = FastAPI(title="CRAG API", version="0.1.0")

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: object,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": "Request validation failed."},
        )

    @app.post(
        "/api/v1/ask",
        response_model=AskResponse,
        status_code=status.HTTP_200_OK,
    )
    async def ask(request: AskRequest) -> AskResponse:
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
                content={
                    "detail": "The CRAG workflow could not produce an answer.",
                    "diagnostics": _safe_diagnostics(state),
                },
            )
        return AskResponse(
            answer=state.get("final_answer"),
            sources=_build_sources(state),
            fallback_used=state.get("fallback_used", False),
            diagnostics=_safe_diagnostics(state),
        )

    @app.get("/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        health: HealthStatus = get_health_status()
        return HealthResponse(
            status="healthy",
            service=health.service,
            checks=health.checks,
        )

    @app.get("/health/ready", response_model=HealthResponse)
    async def ready() -> HealthResponse:
        return HealthResponse(
            status="ready",
            checks={"application": True, "workflow_configured": True},
        )

    return app


def _build_sources(state: CragState) -> list[SourceReference]:
    sources: list[SourceReference] = []
    decisions = state.get("grading_decisions", [])
    for index, evidence in enumerate(state.get("retrieved_evidence", [])):
        if index < len(decisions) and decisions[index].decision == "yes":
            sources.append(
                SourceReference(
                    kind="internal",
                    source_id=evidence.source_id,
                    metadata=_safe_metadata(evidence.metadata),
                )
            )
    for result in state.get("web_results", []):
        sources.append(
            SourceReference(
                kind="external",
                title=result.title,
                url=str(result.url),
            )
        )
    return sources


def _safe_diagnostics(state: CragState) -> list[str]:
    diagnostics: list[str] = []
    for diagnostic in state.get("diagnostics", []):
        if diagnostic.startswith("Internal retrieval failed"):
            diagnostics.append("internal_retrieval_failed")
        elif diagnostic.startswith("Evidence grading failed"):
            diagnostics.append("evidence_grading_failed")
        elif diagnostic.startswith("Web search failed"):
            diagnostics.append("web_search_failed")
        elif diagnostic.startswith("Answer generation failed"):
            diagnostics.append("answer_generation_failed")
    if state.get("error", "").startswith("Web search failed"):
        diagnostics.append("web_search_failed")
    elif state.get("error", "").startswith("Answer generation failed"):
        diagnostics.append("answer_generation_failed")
    return list(dict.fromkeys(diagnostics))


def _safe_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Redact values under common secret-bearing metadata keys."""
    safe: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        normalized = key.lower().replace("-", "_")
        if any(
            marker in normalized
            for marker in ("secret", "password", "token", "api_key", "credential")
        ):
            safe[key] = "[REDACTED]"
        else:
            safe[key] = value
    return safe
