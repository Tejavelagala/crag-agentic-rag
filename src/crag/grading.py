"""Backward-compatible grading imports."""

from crag.rag.grading import (
    PRODUCTION_PROMPT_VERSION,
    build_grading_prompt,
    production_prompt_hash,
)

__all__ = [
    "PRODUCTION_PROMPT_VERSION",
    "build_grading_prompt",
    "production_prompt_hash",
]
