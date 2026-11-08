"""Offline evaluation, benchmarking, and report publication."""

from crag.evaluation.runner import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationDataset,
    EvaluationDocument,
    EvaluationReport,
    _case_assessment,
    _close_resources,
    _fact_supported,
    _owned_point_ids,
    aggregate_metrics,
    load_dataset,
    run_evaluation,
    write_report,
)

__all__ = [
    "EvaluationCase",
    "EvaluationCaseResult",
    "EvaluationDataset",
    "EvaluationDocument",
    "EvaluationReport",
    "aggregate_metrics",
    "load_dataset",
    "run_evaluation",
    "write_report",
    "_case_assessment",
    "_close_resources",
    "_fact_supported",
    "_owned_point_ids",
]
