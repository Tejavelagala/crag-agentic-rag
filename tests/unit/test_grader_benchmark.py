import json
from pathlib import Path

import pytest

from crag.grader_benchmark import (
    EXPERIMENTAL_PROMPT_VERSION,
    PRODUCTION_PROMPT_VERSION,
    BenchmarkCase,
    BenchmarkResult,
    load_benchmark,
    summarize,
)
from crag.grading import build_grading_prompt


def test_fixed_grader_benchmark_has_balanced_cases() -> None:
    benchmark = load_benchmark(Path("evaluation/grader_benchmark.json"))

    assert len(benchmark.cases) == 32
    assert sum(case.expected_decision == "yes" for case in benchmark.cases) == 16
    assert sum(case.expected_decision == "no" for case in benchmark.cases) == 16


def test_grader_benchmark_summary_calculates_error_rates() -> None:
    results = [
        BenchmarkResult(
            case_id="tp",
            category="direct",
            expected_decision="yes",
            actual_decision="yes",
            latency_ms=10,
        ),
        BenchmarkResult(
            case_id="tn",
            category="direct",
            expected_decision="no",
            actual_decision="no",
            latency_ms=20,
        ),
        BenchmarkResult(
            case_id="fp",
            category="irrelevant",
            expected_decision="no",
            actual_decision="yes",
            latency_ms=30,
        ),
        BenchmarkResult(
            case_id="fn",
            category="paraphrase",
            expected_decision="yes",
            actual_decision="no",
            latency_ms=40,
        ),
        BenchmarkResult(
            case_id="invalid",
            category="direct",
            expected_decision="yes",
            error="ProviderError",
            latency_ms=50,
        ),
    ]

    metrics = summarize(results)

    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["invalid_structured_output"] == 1
    assert metrics["false_rejection_rate"] == 0.5
    assert metrics["false_acceptance_rate"] == 0.5


def test_benchmark_rejects_unknown_prompt_variant() -> None:
    assert PRODUCTION_PROMPT_VERSION != EXPERIMENTAL_PROMPT_VERSION
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        from crag.grader_benchmark import _prompt

        _prompt(
            BenchmarkCase(
                case_id="case",
                category="direct",
                question="Question?",
                evidence="Evidence.",
                expected_decision="yes",
                expected_rationale="Relevant.",
            ),
            "unknown",
        )


def test_benchmark_production_prompt_matches_workflow_prompt() -> None:
    case = BenchmarkCase(
        case_id="case",
        category="direct",
        question="Which endpoint reports application liveness?",
        evidence="The endpoint is /health/live.",
        expected_decision="yes",
        expected_rationale="Relevant.",
    )

    from crag.grader_benchmark import _prompt

    assert _prompt(case, PRODUCTION_PROMPT_VERSION) == build_grading_prompt(
        case.question, case.evidence
    )


def test_benchmark_loader_rejects_unbalanced_labels(tmp_path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "cases": [
                    {
                        "case_id": f"case-{index}",
                        "category": "direct",
                        "question": "Question?",
                        "evidence": "Evidence.",
                        "expected_decision": "yes",
                        "expected_rationale": "Relevant.",
                    }
                    for index in range(30)
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="both yes and no"):
        load_benchmark(path)
