"""Backward-compatible benchmark imports and module entry point."""

from crag.evaluation.benchmark import (
    EXPERIMENTAL_PROMPT_VERSION,
    PRODUCTION_PROMPT_VERSION,
    Benchmark,
    BenchmarkCase,
    BenchmarkResult,
    _prompt,
    load_benchmark,
    main,
    summarize,
)

__all__ = [
    "EXPERIMENTAL_PROMPT_VERSION",
    "PRODUCTION_PROMPT_VERSION",
    "Benchmark",
    "BenchmarkCase",
    "BenchmarkResult",
    "load_benchmark",
    "main",
    "summarize",
    "_prompt",
]

if __name__ == "__main__":
    main()
