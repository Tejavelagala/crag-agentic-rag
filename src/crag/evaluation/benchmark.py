"""Controlled benchmark for evidence-relevance grading prompt variants."""

import argparse
import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from crag.core.config import Settings
from crag.core.exceptions import ProviderError
from crag.core.models import GradingDecision
from crag.providers.base import DEFAULT_PROVIDER_TIMEOUT_SECONDS
from crag.providers.ollama import OllamaChatProvider
from crag.rag.grading import (
    PRODUCTION_PROMPT_VERSION,
    build_grading_prompt,
    production_prompt_hash,
)

DEFAULT_BENCHMARK_PATH = Path("evaluation/grader_benchmark.json")
DEFAULT_OUTPUT_PATH = Path("evaluation/results/grader_benchmark.json")
EXPERIMENTAL_PROMPT_VERSION = "explicit-relevance-v1"


class BenchmarkCase(BaseModel):
    """A fixed, independently labeled grading example."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    question: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    expected_decision: Literal["yes", "no"]
    expected_rationale: str = Field(min_length=1)


class Benchmark(BaseModel):
    """Versioned evidence-grading benchmark."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    cases: list[BenchmarkCase] = Field(min_length=30)


class BenchmarkResult(BaseModel):
    """Safe result for one model and prompt variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    category: str
    expected_decision: Literal["yes", "no"]
    actual_decision: Literal["yes", "no"] | None = None
    rationale: str | None = None
    latency_ms: float | None = None
    error: str | None = None


def load_benchmark(path: Path) -> Benchmark:
    """Load and validate the fixed benchmark."""
    try:
        benchmark = Benchmark.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Unable to load grader benchmark: {path}") from exc
    if len({case.case_id for case in benchmark.cases}) != len(benchmark.cases):
        raise ValueError("Grader benchmark case_id values must be unique")
    label_counts = {
        decision: sum(case.expected_decision == decision for case in benchmark.cases)
        for decision in ("yes", "no")
    }
    if not all(label_counts.values()):
        raise ValueError("Grader benchmark must contain both yes and no labels")
    if label_counts["yes"] != label_counts["no"]:
        raise ValueError("Grader benchmark yes/no labels must be balanced")
    return benchmark


def _prompt(case: BenchmarkCase, variant: str) -> str:
    if variant == PRODUCTION_PROMPT_VERSION:
        return build_grading_prompt(case.question, case.evidence)
    elif variant == EXPERIMENTAL_PROMPT_VERSION:
        instruction = (
            "Decide whether the evidence materially helps answer the question. "
            "Use yes only when the evidence directly answers the question or "
            "provides a necessary part of a complete answer. Use no when it is "
            "only topically related, answers a different question, or answers "
            "only one part of a multi-part question. Return a structured "
            "yes/no decision and a concise rationale."
        )
    else:
        raise ValueError(f"Unknown prompt variant: {variant}")
    return f"{instruction}\nQuestion: {case.question}\nEvidence: {case.evidence}"


async def run_variant(
    benchmark: Benchmark,
    *,
    settings: Settings,
    model: str,
    variant: str,
) -> list[BenchmarkResult]:
    """Run every fixed case with one provider/model/prompt variant."""
    provider = OllamaChatProvider(settings, model=model)
    results: list[BenchmarkResult] = []
    try:
        for case in benchmark.cases:
            started = time.monotonic()
            try:
                decision = await provider.generate_structured(
                    _prompt(case, variant),
                    GradingDecision,
                    timeout=DEFAULT_PROVIDER_TIMEOUT_SECONDS,
                )
                results.append(
                    BenchmarkResult(
                        case_id=case.case_id,
                        category=case.category,
                        expected_decision=case.expected_decision,
                        actual_decision=decision.decision,
                        rationale=decision.rationale,
                        latency_ms=(time.monotonic() - started) * 1_000,
                    )
                )
            except (ProviderError, ValueError, TypeError) as exc:
                results.append(
                    BenchmarkResult(
                        case_id=case.case_id,
                        category=case.category,
                        expected_decision=case.expected_decision,
                        latency_ms=(time.monotonic() - started) * 1_000,
                        error=type(exc).__name__,
                    )
                )
    finally:
        await provider.close()
    return results


def summarize(results: list[BenchmarkResult]) -> dict[str, object]:
    """Compute confusion-matrix and latency metrics."""
    tp = sum(
        result.expected_decision == result.actual_decision == "yes"
        for result in results
    )
    tn = sum(
        result.expected_decision == result.actual_decision == "no" for result in results
    )
    fp = sum(
        result.expected_decision == "no" and result.actual_decision == "yes"
        for result in results
    )
    fn = sum(
        result.expected_decision == "yes" and result.actual_decision == "no"
        for result in results
    )
    invalid = sum(result.error is not None for result in results)
    valid = len(results) - invalid
    relevant = tp + fn
    irrelevant = tn + fp
    latencies = [
        result.latency_ms for result in results if result.latency_ms is not None
    ]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / relevant if relevant else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "total_cases": len(results),
        "valid_cases": valid,
        "invalid_structured_output": invalid,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision_relevant": precision,
        "recall_relevant": recall,
        "f1_relevant": f1,
        "false_rejection_rate": fn / relevant if relevant else None,
        "false_acceptance_rate": fp / irrelevant if irrelevant else None,
        "mean_latency_ms": statistics.mean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "min_latency_ms": min(latencies) if latencies else None,
        "max_latency_ms": max(latencies) if latencies else None,
    }


async def run_benchmark(
    benchmark: Benchmark,
    *,
    settings: Settings,
    model: str,
) -> dict[str, object]:
    """Run production and experimental prompts on identical cases."""
    variants = {}
    for variant in (PRODUCTION_PROMPT_VERSION, EXPERIMENTAL_PROMPT_VERSION):
        results = await run_variant(
            benchmark,
            settings=settings,
            model=model,
            variant=variant,
        )
        variants[variant] = {
            "model": model,
            "results": [result.model_dump(mode="json") for result in results],
            "metrics": summarize(results),
        }
    return {
        "benchmark_version": benchmark.version,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "production_prompt_version": PRODUCTION_PROMPT_VERSION,
        "production_prompt_hash": production_prompt_hash(),
        "prompt_variants": variants,
        "limitations": [
            "Labels are fixed human-authored benchmark labels, not model judgments.",
            "No alternative compatible local chat model was available.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=Settings().ollama_chat_model)
    args = parser.parse_args()
    benchmark = load_benchmark(args.benchmark)
    report = asyncio.run(
        run_benchmark(benchmark, settings=Settings(), model=args.model)
    )
    from crag.evaluation.reporting import write_json_report

    write_json_report(report, args.output)
    print(
        json.dumps(
            {key: value["metrics"] for key, value in report["prompt_variants"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
