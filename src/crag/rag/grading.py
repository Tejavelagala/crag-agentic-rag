"""Shared structured evidence-grading prompt construction."""

import hashlib

PRODUCTION_PROMPT_VERSION = "semantic-relevance-v2"
PRODUCTION_PROMPT = (
    "Decide yes if the evidence contains information capable of answering or "
    "materially supporting an answer to the question. Semantic paraphrases count "
    "as relevant; exact keyword matching is not required. Decide no when the "
    "evidence is unrelated or insufficient. Do not invent or infer facts absent "
    "from the evidence. Respond with a structured yes/no decision and rationale."
)


def build_grading_prompt(question: str, evidence: str) -> str:
    """Build the canonical production grading prompt."""
    return f"{PRODUCTION_PROMPT}\nQuestion: {question}\nEvidence: {evidence}"


def production_prompt_hash() -> str:
    """Return a stable identifier for the canonical production prompt."""
    return hashlib.sha256(PRODUCTION_PROMPT.encode("utf-8")).hexdigest()
