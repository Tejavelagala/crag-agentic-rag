"""Tests for the deterministic evaluation dataset and report helpers."""

import asyncio
import json

import httpx
import pytest

from crag.evaluation import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    _case_assessment,
    _close_resources,
    _fact_supported,
    aggregate_metrics,
    load_dataset,
    write_report,
)
from crag.reporting import write_json_report


def test_evaluation_dataset_validates_sources_and_expectations() -> None:
    dataset = EvaluationDataset(
        version="test",
        documents=[{"document_id": "doc-1", "text": "Fact"}],
        cases=[
            {
                "case_id": "case-1",
                "question": "What fact?",
                "expected_facts": ["Fact"],
                "relevant_source_ids": ["doc-1"],
                "sufficient_internal_evidence": True,
                "expected_fallback": False,
            }
        ],
    )

    assert dataset.cases[0].relevant_source_ids == ["doc-1"]

    with pytest.raises(ValueError, match="unknown source"):
        EvaluationDataset(
            version="test",
            documents=[{"document_id": "doc-1", "text": "Fact"}],
            cases=[
                {
                    "case_id": "case-1",
                    "question": "What fact?",
                    "relevant_source_ids": ["missing"],
                    "sufficient_internal_evidence": True,
                    "expected_fallback": False,
                }
            ],
        )


def test_evaluation_metrics_and_output_are_deterministic(tmp_path) -> None:
    results = [
        EvaluationCaseResult(
            case_id="supported",
            status_code=200,
            answer="Fact",
            source_ids=["doc-1"],
            latency_ms=10,
            passed=True,
        ),
        EvaluationCaseResult(
            case_id="unsupported",
            status_code=503,
            diagnostics=["web_search_failed"],
            latency_ms=20,
            passed=True,
        ),
    ]

    metrics = aggregate_metrics(results)
    assert metrics["passed_cases"] == 2
    assert metrics["fallback_rate"] == 0.5
    assert metrics["mean_latency_ms"] == 15
    assert metrics["median_latency_ms"] == 15

    report_path = tmp_path / "nested" / "report.json"
    from crag.evaluation import EvaluationReport

    report = EvaluationReport(
        dataset_version="test",
        evaluated_at="2026-01-01T00:00:00Z",
        provider_mode="local",
        chat_model="chat",
        embedding_model="embedding",
        collection_name="crag_evaluation_test",
        cases=results,
        metrics=metrics,
        limitations=[],
    )
    write_report(report, report_path)
    assert json.loads(report_path.read_text())["metrics"]["passed_cases"] == 2


def test_dataset_loader_rejects_malformed_file(tmp_path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text('{"version": "bad"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to load evaluation dataset"):
        load_dataset(path)


def test_report_writers_preserve_previous_report(tmp_path) -> None:
    path = tmp_path / "latest.json"
    path.write_text('{"evaluated_at":"old"}\n', encoding="utf-8")

    write_json_report({"evaluated_at": "new"}, path)

    assert json.loads(path.read_text())["evaluated_at"] == "new"
    history = list((tmp_path / "history").glob("latest-*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text())["evaluated_at"] == "old"


@pytest.mark.parametrize(
    ("expected_fact", "answer"),
    [
        (
            "retrieves internal evidence",
            "The workflow first retrieves internal evidence.",
        ),
        ("grades its relevance", "The evidence is graded for relevance."),
        ("does not invent", "It does not generate or invent results."),
    ],
)
def test_fact_matching_accepts_defensible_paraphrases(
    expected_fact: str,
    answer: str,
) -> None:
    assert _fact_supported(expected_fact, answer)


def test_metrics_exclude_fallbacks_from_answer_quality_rates() -> None:
    results = [
        EvaluationCaseResult(
            case_id="supported",
            status_code=200,
            answer="Fact",
            context_relevant=True,
            faithful=True,
            answer_relevant=True,
            passed=True,
        ),
        EvaluationCaseResult(
            case_id="fallback",
            status_code=503,
            diagnostics=["web_search_failed"],
            context_relevant=False,
        ),
    ]

    metrics = aggregate_metrics(results)

    assert metrics["context_relevance_rate"] == 0.5
    assert metrics["faithfulness_rate"] == 1.0
    assert metrics["answer_relevance_rate"] == 1.0


def test_unexpected_http_errors_are_counted_but_expected_fallbacks_are_not() -> None:
    unexpected = EvaluationCaseResult(
        case_id="unexpected",
        status_code=500,
        error="http_status_500",
    )
    expected = EvaluationCaseResult(
        case_id="expected",
        status_code=503,
        diagnostics=["web_search_failed"],
        passed=True,
    )

    metrics = aggregate_metrics([unexpected, expected])

    assert metrics["error_rate"] == 0.5
    assert metrics["passed_cases"] == 1


def test_case_assessment_preserves_expected_fallback_as_success() -> None:
    case = EvaluationCase(
        case_id="fallback",
        question="Unknown?",
        expected_facts=[],
        relevant_source_ids=[],
        sufficient_internal_evidence=False,
        expected_fallback=True,
    )
    response = httpx.Response(
        503,
        json={"diagnostics": ["web_search_failed"], "answer": None},
    )

    passed, context_relevant, faithful, answer_relevant = _case_assessment(
        case, response
    )

    assert passed is True
    assert context_relevant is False
    assert faithful is None
    assert answer_relevant is None


def test_evaluation_cleanup_attempts_all_resource_closes() -> None:
    class Resource:
        def __init__(self, fails: bool = False) -> None:
            self.closed = False
            self.fails = fails

        async def close(self) -> None:
            self.closed = True
            if self.fails:
                raise RuntimeError("close failed")

    failing = Resource(fails=True)
    succeeding = Resource()
    error = asyncio.run(_close_resources((failing, succeeding)))

    assert isinstance(error, RuntimeError)
    assert failing.closed is True
    assert succeeding.closed is True
