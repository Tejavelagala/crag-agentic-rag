"""Reproducible evaluation runner for the local CRAG application."""

import argparse
import asyncio
import json
import os
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import AsyncQdrantClient, models

from crag.api import create_app
from crag.core.config import Settings
from crag.ingestion import ingest_document
from crag.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from crag.providers.qdrant import QdrantVectorStoreProvider
from crag.rag.workflow import CragWorkflow
from crag.services.composition import DisabledWebSearchProvider

DEFAULT_DATASET_PATH = Path("evaluation/dataset.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/latest.json")
EVALUATION_COLLECTION_ENV = "CRAG_EVALUATION_QDRANT_COLLECTION"


class EvaluationDocument(BaseModel):
    """A fixed document inserted for one evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvaluationCase(BaseModel):
    """A deterministic evaluation question and expected behavior."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_facts: list[str] = Field(default_factory=list)
    relevant_source_ids: list[str] = Field(default_factory=list)
    sufficient_internal_evidence: bool
    expected_fallback: bool

    @model_validator(mode="after")
    def validate_expectations(self) -> "EvaluationCase":
        if self.sufficient_internal_evidence and not self.relevant_source_ids:
            raise ValueError("supported cases require relevant_source_ids")
        if not self.sufficient_internal_evidence and self.expected_facts:
            raise ValueError("unsupported cases must not define expected facts")
        return self

    @property
    def relevant_facts(self) -> list[str]:
        return self.expected_facts


class EvaluationDataset(BaseModel):
    """Versioned, secret-free evaluation corpus and cases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    documents: list[EvaluationDocument] = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sources(self) -> "EvaluationDataset":
        document_ids = {document.document_id for document in self.documents}
        if len(document_ids) != len(self.documents):
            raise ValueError("document_id values must be unique")
        for case in self.cases:
            if not set(case.relevant_source_ids).issubset(document_ids):
                raise ValueError(f"case {case.case_id} references an unknown source")
        return self


class EvaluationCaseResult(BaseModel):
    """Machine-readable result for one evaluation case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

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


class EvaluationReport(BaseModel):
    """Evaluation output with transparent deterministic metrics."""

    model_config = ConfigDict(extra="forbid")

    dataset_version: str
    evaluated_at: datetime
    provider_mode: str
    chat_model: str
    embedding_model: str
    collection_name: str
    cases: list[EvaluationCaseResult]
    metrics: dict[str, float | int | None]
    limitations: list[str]


def load_dataset(path: Path) -> EvaluationDataset:
    """Load and validate a version-controlled dataset."""
    try:
        return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Unable to load evaluation dataset: {path}") from exc


def aggregate_metrics(
    results: list[EvaluationCaseResult],
) -> dict[str, float | int | None]:
    """Aggregate deterministic outcome and latency metrics."""
    completed = [result for result in results if result.status_code is not None]
    latencies = [
        result.latency_ms for result in completed if result.latency_ms is not None
    ]
    fallback_count = sum(
        result.fallback_used or "web_search_failed" in result.diagnostics
        for result in completed
    )
    context_results = [
        result.context_relevant
        for result in completed
        if result.context_relevant is not None
    ]
    faithfulness_results = [
        result.faithful for result in completed if result.faithful is not None
    ]
    relevance_results = [
        result.answer_relevant
        for result in completed
        if result.answer_relevant is not None
    ]
    return {
        "total_cases": len(results),
        "completed_cases": len(completed),
        "skipped_cases": len(results) - len(completed),
        "passed_cases": sum(result.passed for result in completed),
        "answer_success_rate": _rate(
            sum(result.passed for result in completed),
            len(completed),
        ),
        "context_relevance_rate": _boolean_rate(context_results),
        "faithfulness_rate": _boolean_rate(faithfulness_results),
        "answer_relevance_rate": _boolean_rate(relevance_results),
        "fallback_rate": _rate(fallback_count, len(completed)),
        "error_rate": _rate(
            sum(bool(result.error) for result in completed),
            len(completed),
        ),
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _boolean_rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return _rate(sum(usable), len(usable))


def _case_assessment(
    case: EvaluationCase,
    response: httpx.Response,
    retrieved_source_ids: set[str] | None = None,
) -> tuple[bool, bool, bool | None, bool | None]:
    body = response.json()
    answer = body.get("answer")
    answer_text = answer if isinstance(answer, str) else ""
    source_ids = {
        source.get("source_id")
        for source in body.get("sources", [])
        if source.get("kind") == "internal"
    }
    context_ids = (
        retrieved_source_ids if retrieved_source_ids is not None else source_ids
    )
    context_relevant = bool(context_ids.intersection(case.relevant_source_ids))
    has_answer = response.status_code == 200 and bool(answer_text.strip())
    faithful = (
        all(_fact_supported(fact, answer_text) for fact in case.expected_facts)
        if has_answer
        else None
    )
    answer_relevant = faithful
    if case.expected_fallback:
        passed = response.status_code == 503 and "web_search_failed" in body.get(
            "diagnostics", []
        )
    else:
        passed = (
            response.status_code == 200
            and bool(answer_text.strip())
            and context_relevant
            and faithful
        )
    return passed, context_relevant, faithful, answer_relevant


async def _close_resources(resources: tuple[object, ...]) -> Exception | None:
    first_error: Exception | None = None
    for resource in resources:
        close = getattr(resource, "close", None)
        if close is None:
            continue
        try:
            await close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    return first_error


async def _owned_point_ids(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    run_id: str,
    dataset_version: str,
) -> set[str]:
    owned: set[str] = set()
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection_name,
            limit=10_000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = getattr(record, "payload", {}) or {}
            metadata = payload.get("metadata", {})
            if (
                isinstance(metadata, dict)
                and metadata.get("evaluation_run_id") == run_id
                and metadata.get("dataset_version") == dataset_version
            ):
                owned.add(str(getattr(record, "id", "")))
        if offset is None:
            break
    return owned


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "its",
    "of",
    "the",
    "to",
}


def _fact_supported(expected_fact: str, answer: str) -> bool:
    """Match ordered content words with simple morphological normalization."""
    expected_tokens = [
        _normalize_token(token)
        for token in re.findall(r"[a-z0-9]+", expected_fact.lower())
        if token not in _STOP_WORDS
    ]
    answer_tokens = [
        _normalize_token(token) for token in re.findall(r"[a-z0-9]+", answer.lower())
    ]
    if not expected_tokens:
        return True
    position = 0
    for token in answer_tokens:
        if _tokens_match(token, expected_tokens[position]):
            position += 1
            if position == len(expected_tokens):
                return True
    return False


def _tokens_match(left: str, right: str) -> bool:
    return left == right or left == right.rstrip("e") or right == left.rstrip("e")


def _normalize_token(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


class _RecordingWorkflow:
    """Capture safe workflow state for evaluation diagnostics."""

    def __init__(self, workflow: CragWorkflow) -> None:
        self._workflow = workflow
        self.state = None

    async def invoke(self, question: str):
        self.state = await self._workflow.invoke(question)
        return self.state


async def run_evaluation(
    dataset: EvaluationDataset,
    *,
    settings: Settings,
    collection_name: str,
) -> EvaluationReport:
    """Insert evaluation data, exercise the real API, and clean up its points."""
    if not collection_name.startswith("crag_evaluation_"):
        raise ValueError(
            "Evaluation requires a dedicated collection beginning with "
            "'crag_evaluation_'"
        )
    if not settings.qdrant_url:
        raise ValueError("CRAG_QDRANT_URL is required for evaluation")

    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )
    embeddings = OllamaEmbeddingProvider(settings)
    chat = OllamaChatProvider(settings)
    vector_store: QdrantVectorStoreProvider | None = None
    workflow: CragWorkflow | None = None
    point_ids: list[str] = []
    evaluation_run_id = f"evaluation-{uuid4().hex}"
    try:
        vector = await embeddings.embed_query("CRAG evaluation dimension probe")
        vector_store = QdrantVectorStoreProvider(
            settings,
            embeddings,
            collection_name=collection_name,
            vector_size=len(vector),
            client=client,
        )
        workflow = CragWorkflow(
            vector_store=vector_store,
            chat=chat,
            web_search=DisabledWebSearchProvider(),
            top_k=1,
            web_search_enabled=False,
        )
        chunks = [
            chunk.model_copy(
                update={
                    "chunk_id": f"{evaluation_run_id}:{chunk.chunk_id}",
                    "parent_id": (
                        f"{evaluation_run_id}:{chunk.parent_id}"
                        if chunk.parent_id is not None
                        else None
                    ),
                }
            )
            for document in dataset.documents
            for chunk in ingest_document(
                document.text,
                document_id=document.document_id,
                metadata={
                    "evaluation": True,
                    "evaluation_run_id": evaluation_run_id,
                    "dataset_version": dataset.version,
                },
                parent_size=2_000,
                child_size=500,
                overlap=50,
            )
        ]
        point_ids = [str(uuid5(NAMESPACE_URL, chunk.chunk_id)) for chunk in chunks]
        await vector_store.upsert_chunks(chunks)
        recording_workflow = _RecordingWorkflow(workflow)
        app = create_app(recording_workflow)
        results: list[EvaluationCaseResult] = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://evaluation",
        ) as http_client:
            for case in dataset.cases:
                started = time.monotonic()
                try:
                    response = await http_client.post(
                        "/api/v1/ask",
                        json={"question": case.question},
                    )
                    body = response.json()
                    state = recording_workflow.state or {}
                    retrieved_source_ids = {
                        evidence.source_id
                        for evidence in state.get("retrieved_evidence", [])
                    }
                    (
                        passed,
                        context_relevant,
                        faithful,
                        answer_relevant,
                    ) = _case_assessment(case, response, retrieved_source_ids)
                    diagnostics = body.get("diagnostics", [])
                    expected_fallback_response = (
                        case.expected_fallback
                        and response.status_code == 503
                        and "web_search_failed" in diagnostics
                    )
                    response_error = (
                        None
                        if response.status_code == 200 or expected_fallback_response
                        else f"http_status_{response.status_code}"
                    )
                    results.append(
                        EvaluationCaseResult(
                            case_id=case.case_id,
                            status_code=response.status_code,
                            answer=body.get("answer"),
                            source_ids=[
                                source["source_id"]
                                for source in body.get("sources", [])
                                if source.get("source_id")
                            ],
                            retrieved_source_ids=[
                                evidence.source_id
                                for evidence in state.get("retrieved_evidence", [])
                            ],
                            grading_decisions=[
                                decision.decision
                                for decision in state.get("grading_decisions", [])
                            ],
                            fallback_used=body.get("fallback_used", False)
                            or "web_search_failed" in diagnostics,
                            diagnostics=diagnostics,
                            stage_timings_ms=state.get("stage_timings_ms", {}),
                            latency_ms=(time.monotonic() - started) * 1_000,
                            context_relevant=context_relevant,
                            faithful=faithful,
                            answer_relevant=answer_relevant,
                            passed=passed,
                            error=response_error,
                        )
                    )
                except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                    results.append(
                        EvaluationCaseResult(
                            case_id=case.case_id,
                            latency_ms=(time.monotonic() - started) * 1_000,
                            error=type(exc).__name__,
                        )
                    )
        return EvaluationReport(
            dataset_version=dataset.version,
            evaluated_at=datetime.now(UTC),
            provider_mode=settings.provider_mode,
            chat_model=settings.ollama_chat_model,
            embedding_model=settings.ollama_embedding_model,
            collection_name=collection_name,
            cases=results,
            metrics=aggregate_metrics(results),
            limitations=[
                "Faithfulness and answer relevance are deterministic "
                "expected-fact heuristics, not RAGAS scores.",
                "No hosted web fallback was contacted in local-only mode.",
            ],
        )
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_error: Exception | None = None
        try:
            if point_ids:
                owned_ids = await _owned_point_ids(
                    client,
                    collection_name,
                    run_id=evaluation_run_id,
                    dataset_version=dataset.version,
                )
                cleanup_ids = [
                    point_id for point_id in point_ids if point_id in owned_ids
                ]
                if cleanup_ids:
                    await client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(points=cleanup_ids),
                        wait=True,
                    )
        except Exception as exc:
            cleanup_error = exc
        finally:
            workflow_resources = (
                (workflow,) if workflow is not None else (vector_store,)
            )
            close_error = await _close_resources(
                workflow_resources + (embeddings, chat, client)
            )
        if primary_error is None:
            if cleanup_error is not None:
                raise cleanup_error
            if close_error is not None:
                raise close_error


def write_report(report: EvaluationReport, output_path: Path) -> None:
    """Archive an existing report before publishing the new report."""
    from crag.evaluation.reporting import write_json_report

    write_json_report(report.model_dump(mode="json"), output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--collection",
        default=os.getenv(EVALUATION_COLLECTION_ENV, ""),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.collection:
        raise SystemExit(
            f"Set {EVALUATION_COLLECTION_ENV} to a dedicated evaluation collection."
        )
    dataset = load_dataset(args.dataset)
    settings = Settings()
    report = asyncio.run(
        run_evaluation(dataset, settings=settings, collection_name=args.collection)
    )
    write_report(report, args.output)
    print(json.dumps(report.metrics, indent=2))


if __name__ == "__main__":
    main()
