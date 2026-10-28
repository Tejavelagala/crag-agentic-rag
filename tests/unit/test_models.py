"""Core data model behavior tests."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from crag.models import (
    DocumentChunk,
    GradingDecision,
    RetrievedEvidence,
    WebSearchResult,
)


def test_document_chunk_accepts_json_metadata_and_independent_defaults() -> None:
    first = DocumentChunk(chunk_id="chunk-1", document_id="doc-1", text="Content")
    second = DocumentChunk(chunk_id="chunk-2", document_id="doc-2", text="Other")

    first.metadata["page"] = 1

    assert first.metadata == {"page": 1}
    assert second.metadata == {}


@pytest.mark.parametrize(
    "model",
    [
        lambda: DocumentChunk(chunk_id=" ", document_id="doc", text="content"),
        lambda: DocumentChunk(chunk_id="chunk", document_id=" ", text="content"),
        lambda: DocumentChunk(chunk_id="chunk", document_id="doc", text=" "),
        lambda: RetrievedEvidence(text="", source_id="source"),
        lambda: RetrievedEvidence(text="evidence", source_id="\t"),
        lambda: WebSearchResult(title=" ", url="https://example.com", snippet="text"),
        lambda: WebSearchResult(title="title", url="https://example.com", snippet=" "),
    ],
)
def test_models_reject_blank_required_text(model: object) -> None:
    with pytest.raises(ValidationError):
        model()


def test_retrieved_evidence_accepts_optional_score_and_metadata() -> None:
    evidence = RetrievedEvidence(
        text="Relevant passage",
        source_id="doc-1",
        metadata={"section": "intro"},
        retrieval_score=0.87,
    )

    assert evidence.retrieval_score == 0.87
    assert evidence.metadata == {"section": "intro"}


def test_web_search_result_accepts_http_url_and_publication_time() -> None:
    result = WebSearchResult(
        title="A result",
        url="https://example.com/article",
        snippet="A useful result.",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert str(result.url) == "https://example.com/article"
    assert result.published_at is not None


@pytest.mark.parametrize("url", ["not-a-url", "ftp://example.com/file"])
def test_web_search_result_rejects_malformed_or_unsupported_url(url: str) -> None:
    with pytest.raises(ValidationError):
        WebSearchResult(title="A result", url=url, snippet="A useful result.")


def test_grading_decision_requires_strict_binary_decision_and_rationale() -> None:
    decision = GradingDecision(
        decision="yes", rationale="The passage answers the query."
    )

    assert decision.decision == "yes"

    with pytest.raises(ValidationError):
        GradingDecision(decision="maybe", rationale="Uncertain.")
    with pytest.raises(ValidationError):
        GradingDecision(decision="no", rationale="  ")
