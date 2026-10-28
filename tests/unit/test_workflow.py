"""Deterministic corrective retrieval workflow tests."""

import asyncio

import pytest

from crag.exceptions import ProviderError
from crag.models import GradingDecision, RetrievedEvidence, WebSearchResult
from crag.workflow import invoke_crag


class FakeVectorStore:
    def __init__(
        self,
        evidence: list[RetrievedEvidence] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.evidence = evidence or []
        self.error = error

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        timeout: float = 30.0,
    ) -> list[RetrievedEvidence]:
        if self.error:
            raise self.error
        return self.evidence[:top_k]


class FakeChat:
    def __init__(
        self,
        decisions: list[GradingDecision] | None = None,
        error: Exception | None = None,
        structured_error: Exception | None = None,
    ) -> None:
        self.decisions = decisions or []
        self.error = error
        self.structured_error = structured_error
        self.structured_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    async def generate_structured(
        self,
        prompt: str,
        output_type: type[GradingDecision],
        *,
        timeout: float = 30.0,
    ) -> GradingDecision:
        self.structured_prompts.append(prompt)
        if self.structured_error:
            raise self.structured_error
        if self.error:
            raise self.error
        return self.decisions.pop(0)

    async def generate(
        self,
        prompt: str,
        *,
        timeout: float = 30.0,
    ) -> str:
        self.answer_prompts.append(prompt)
        if self.error:
            raise self.error
        return "grounded answer"


class FakeWebSearch:
    def __init__(
        self,
        results: list[WebSearchResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.calls = 0

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        timeout: float = 30.0,
    ) -> list[WebSearchResult]:
        self.calls += 1
        if self.error:
            raise self.error
        return self.results[:max_results]


def result(title: str = "External") -> WebSearchResult:
    return WebSearchResult(
        title=title,
        url="https://example.com",
        snippet="External evidence",
    )


def test_relevant_internal_evidence_generates_grounded_answer() -> None:
    chat = FakeChat([GradingDecision(decision="yes", rationale="Relevant")])

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "What is CRAG?",
            vector_store=FakeVectorStore(
                [RetrievedEvidence(text="Internal evidence", source_id="doc-1")]
            ),
            chat=chat,
            web_search=FakeWebSearch([result()]),
        )

    state = asyncio.run(exercise())

    assert state["final_answer"] == "grounded answer"
    assert state["fallback_used"] is False
    assert state["web_results"] == []
    assert "Internal evidence" in chat.answer_prompts[0]


def test_irrelevant_internal_evidence_routes_to_web_fallback() -> None:
    chat = FakeChat([GradingDecision(decision="no", rationale="Not relevant")])
    web = FakeWebSearch([result()])

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Question",
            vector_store=FakeVectorStore(
                [RetrievedEvidence(text="Irrelevant", source_id="doc-1")]
            ),
            chat=chat,
            web_search=web,
            max_web_results=1,
        )

    state = asyncio.run(exercise())

    assert state["fallback_used"] is True
    assert state["web_results"] == [result()]
    assert state["final_answer"] == "grounded answer"
    assert web.calls == 1
    assert "External evidence" in chat.answer_prompts[0]


@pytest.mark.parametrize("evidence", [[], None])
def test_empty_retrieval_routes_to_web_without_grading(
    evidence: list[RetrievedEvidence] | None,
) -> None:
    chat = FakeChat()
    web = FakeWebSearch([result()])

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Question",
            vector_store=FakeVectorStore(evidence),
            chat=chat,
            web_search=web,
        )

    state = asyncio.run(exercise())

    assert state["fallback_used"] is True
    assert state["final_answer"] == "grounded answer"
    assert chat.structured_prompts == []


def test_retrieval_failure_is_diagnostic_and_recoverable() -> None:
    web = FakeWebSearch([result()])

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Question",
            vector_store=FakeVectorStore(error=ProviderError("vector unavailable")),
            chat=FakeChat(),
            web_search=web,
        )

    state = asyncio.run(exercise())

    assert state["fallback_used"] is True
    assert state["final_answer"] == "grounded answer"
    assert any("Internal retrieval failed" in item for item in state["diagnostics"])


def test_web_failure_terminates_without_fabricated_answer() -> None:
    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Question",
            vector_store=FakeVectorStore(),
            chat=FakeChat(),
            web_search=FakeWebSearch(error=ProviderError("web unavailable")),
        )

    state = asyncio.run(exercise())

    assert state["final_answer"] is None
    assert state["error"] == "Web search failed: web unavailable"
    assert state["fallback_used"] is True


def test_grading_failure_routes_once_to_web_and_terminates() -> None:
    chat = FakeChat(structured_error=ProviderError("grader unavailable"))
    web = FakeWebSearch([result()])

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Question",
            vector_store=FakeVectorStore(
                [RetrievedEvidence(text="Evidence", source_id="doc-1")]
            ),
            chat=chat,
            web_search=web,
        )

    state = asyncio.run(exercise())

    assert web.calls == 1
    assert state["final_answer"] == "grounded answer"
    assert state["error"] is None


def test_invalid_question_and_workflow_limits_are_rejected() -> None:
    async def exercise() -> None:
        with pytest.raises(ValueError, match="question"):
            await invoke_crag(
                " ",
                vector_store=FakeVectorStore(),
                chat=FakeChat(),
                web_search=FakeWebSearch(),
            )

    asyncio.run(exercise())
