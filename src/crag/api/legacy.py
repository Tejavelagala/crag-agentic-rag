"""FastAPI application for the corrective retrieval workflow."""

import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from crag.core.models import JsonValue
from crag.documents import DocumentChunkRecord, DocumentRecord
from crag.rag.workflow import CragState

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
    retrieved_evidence: list["EvidenceReference"] = Field(default_factory=list)
    grading_decisions: list["GradingReference"] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    stage_statuses: dict[
        str, Literal["completed", "failed", "skipped", "not_reported"]
    ] = Field(default_factory=dict)
    web_search: "WebSearchStatus" = Field(
        default_factory=lambda: WebSearchStatus(
            enabled=False,
            attempted=False,
            status="not_attempted",
        )
    )


class EvidenceReference(BaseModel):
    """Evidence returned by retrieval, without inventing unavailable metadata."""

    source_id: str
    text: str
    retrieval_score: float | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class GradingReference(BaseModel):
    """Structured relevance decision returned by the workflow."""

    decision: Literal["yes", "no"]
    rationale: str


class WebSearchStatus(BaseModel):
    """Safe, explicit status for corrective web fallback."""

    enabled: bool
    attempted: bool
    status: Literal["not_attempted", "disabled", "succeeded", "failed"]
    source_count: int = 0


class AskFailureResponse(BaseModel):
    """Safe response when the workflow cannot produce an answer."""

    detail: str
    diagnostics: list[str] = Field(default_factory=list)
    answer: str | None = None
    sources: list[SourceReference] = Field(default_factory=list)
    fallback_used: bool = False
    retrieved_evidence: list[EvidenceReference] = Field(default_factory=list)
    grading_decisions: list[GradingReference] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    stage_statuses: dict[
        str, Literal["completed", "failed", "skipped", "not_reported"]
    ] = Field(default_factory=dict)
    web_search: WebSearchStatus = Field(
        default_factory=lambda: WebSearchStatus(
            enabled=False,
            attempted=False,
            status="not_attempted",
        )
    )


class EvaluationCaseResponse(BaseModel):
    """Validated read-only evaluation case output."""

    case_id: str
    status_code: int | None = None
    answer: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    retrieved_source_ids: list[str] = Field(default_factory=list)
    grading_decisions: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    diagnostics: list[str] = Field(default_factory=list)
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    latency_ms: float | None = None
    context_relevant: bool | None = None
    faithful: bool | None = None
    answer_relevant: bool | None = None
    passed: bool = False
    error: str | None = None


class EvaluationReportResponse(BaseModel):
    """Validated read-only evaluation report."""

    dataset_version: str
    evaluated_at: str
    provider_mode: str
    chat_model: str
    embedding_model: str
    collection_name: str
    cases: list[EvaluationCaseResponse]
    metrics: dict[str, float | int | None]
    limitations: list[str]


class BenchmarkResultResponse(BaseModel):
    """Validated read-only grader benchmark case output."""

    case_id: str
    category: str
    expected_decision: Literal["yes", "no"]
    actual_decision: Literal["yes", "no"] | None = None
    rationale: str | None = None
    latency_ms: float | None = None
    error: str | None = None


class BenchmarkVariantResponse(BaseModel):
    """Validated results for one benchmark prompt variant."""

    model: str
    results: list[BenchmarkResultResponse]
    metrics: dict[str, float | int | None]


class BenchmarkReportResponse(BaseModel):
    """Validated read-only grader benchmark report."""

    benchmark_version: str
    evaluated_at: str
    model: str
    production_prompt_version: str | None = None
    production_prompt_hash: str | None = None
    prompt_variants: dict[str, BenchmarkVariantResponse]
    limitations: list[str]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_REPORT_PATH = PROJECT_ROOT / "evaluation" / "results" / "latest.json"
BENCHMARK_REPORT_PATH = (
    PROJECT_ROOT / "evaluation" / "results" / "grader_benchmark.json"
)


class HealthResponse(BaseModel):
    """Health response that reports only checks performed by the API."""

    status: Literal["healthy", "ready", "not_ready"]
    service: str = "crag"
    checks: dict[str, bool]


class DocumentPage(BaseModel):
    """Paginated document registry response."""

    items: list[DocumentRecord]
    offset: int
    limit: int
    total: int


class DocumentChunkPage(BaseModel):
    """Paginated chunk inspection response."""

    items: list[DocumentChunkRecord]
    offset: int
    limit: int
    total: int


class DocumentUploadResult(BaseModel):
    """Per-file upload outcome."""

    filename: str
    document: DocumentRecord | None = None
    error: str | None = None
    duplicate_document_id: str | None = None


def create_app(*args: object, **kwargs: object):
    """Compatibility wrapper for the modular API factory."""
    from crag.api import create_app as modular_create_app

    return modular_create_app(*args, **kwargs)


def _read_report(path: Path, model: type[BaseModel]) -> BaseModel:
    """Read one fixed report path and expose only validated report data."""
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The requested evaluation report is not available.",
        ) from exc
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Evaluation report validation failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The requested evaluation report is invalid.",
        ) from exc


def _failure_payload(detail: str, state: CragState) -> dict[str, object]:
    return AskFailureResponse(
        detail=detail,
        diagnostics=_safe_diagnostics(state),
        sources=_build_sources(state),
        fallback_used=state.get("fallback_used", False),
        retrieved_evidence=_build_evidence(state),
        grading_decisions=_build_grading(state),
        retrieved_chunk_count=len(state.get("retrieved_evidence", [])),
        stage_timings_ms=state.get("stage_timings_ms", {}),
        stage_statuses=_stage_statuses(state),
        web_search=_web_search_status(state),
    ).model_dump(mode="json")


def _build_evidence(state: CragState) -> list[EvidenceReference]:
    return [
        EvidenceReference(
            source_id=evidence.source_id,
            text=evidence.text,
            retrieval_score=evidence.retrieval_score,
            metadata=_safe_metadata(evidence.metadata),
        )
        for evidence in state.get("retrieved_evidence", [])
    ]


def _build_grading(state: CragState) -> list[GradingReference]:
    return [
        GradingReference(decision=decision.decision, rationale=decision.rationale)
        for decision in state.get("grading_decisions", [])
    ]


def _stage_statuses(
    state: CragState,
) -> dict[str, Literal["completed", "failed", "skipped", "not_reported"]]:
    timings = state.get("stage_timings_ms", {})
    diagnostics = state.get("diagnostics", [])
    status: dict[str, Literal["completed", "failed", "skipped", "not_reported"]] = {}
    retrieval_failed = any(
        isinstance(item, str) and item.startswith("Internal retrieval failed")
        for item in diagnostics
    )
    grading_failed = any(
        isinstance(item, str) and item.startswith("Evidence grading failed")
        for item in diagnostics
    )
    status["retrieval"] = (
        "failed"
        if retrieval_failed
        else "completed"
        if "retrieval" in timings
        else "not_reported"
    )
    status["grading"] = (
        "failed"
        if grading_failed
        else "completed"
        if "grading" in timings
        else "skipped"
        if not state.get("retrieved_evidence")
        else "not_reported"
    )
    status["routing"] = "completed" if "retrieval" in timings else "not_reported"
    status["answer"] = (
        "failed"
        if state.get("error")
        or any(
            isinstance(item, str) and item.startswith("Answer generation failed")
            for item in diagnostics
        )
        else "completed"
        if "answer_generation" in timings and state.get("final_answer")
        else "skipped"
        if "answer_generation" not in timings
        else "not_reported"
    )
    return status


def _web_search_status(state: CragState) -> WebSearchStatus:
    return WebSearchStatus(
        enabled=state.get("web_search_enabled", False),
        attempted=state.get("web_search_attempted", False),
        status=state.get("web_search_status", "not_attempted"),
        source_count=len(state.get("web_results", [])),
    )


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
    error = state.get("error") or ""
    if error.startswith("Web search failed"):
        diagnostics.append("web_search_failed")
    elif error.startswith("Answer generation failed"):
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
