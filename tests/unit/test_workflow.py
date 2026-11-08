"""Deterministic corrective retrieval workflow tests."""

import asyncio

import pytest

from crag.exceptions import ProviderError
from crag.grading import build_grading_prompt
from crag.models import GradingDecision, RetrievedEvidence, WebSearchResult
from crag.workflow import CragWorkflow, invoke_crag


class FakeVectorStore:
    def __init__(
        self,
        evidence: list[RetrievedEvidence] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.evidence = evidence or []
        self.error = error
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

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
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

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
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

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


def test_additional_vector_store_evidence_is_merged_and_ranked() -> None:
    primary = FakeVectorStore(
        [RetrievedEvidence(source_id="demo", text="demo", retrieval_score=0.4)]
    )
    uploaded = FakeVectorStore(
        [RetrievedEvidence(source_id="upload", text="upload", retrieval_score=0.9)]
    )
    workflow = CragWorkflow(
        vector_store=primary,
        additional_vector_stores=(uploaded,),
        chat=FakeChat(
            [
                GradingDecision(decision="yes", rationale="relevant"),
                GradingDecision(decision="no", rationale="irrelevant"),
            ]
        ),
        web_search=FakeWebSearch(),
        top_k=2,
    )

    state = asyncio.run(workflow.invoke("question"))

    assert [item.source_id for item in state["retrieved_evidence"]] == [
        "upload",
        "demo",
    ]


def test_liveness_grading_prompt_accepts_semantic_endpoint_evidence() -> None:
    chat = FakeChat(
        [
            GradingDecision(
                decision="yes", rationale="It identifies the liveness endpoint."
            )
        ]
    )

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Which endpoint reports application liveness?",
            vector_store=FakeVectorStore(
                [
                    RetrievedEvidence(
                        text=(
                            "The CRAG demo API exposes /health/live for liveness "
                            "and /health/ready for readiness."
                        ),
                        source_id="demo-api-operations",
                    )
                ]
            ),
            chat=chat,
            web_search=FakeWebSearch(),
        )

    state = asyncio.run(exercise())

    assert state["grading_decisions"][0].decision == "yes"
    assert state["final_answer"] == "grounded answer"
    assert "Semantic paraphrases count as relevant" in chat.structured_prompts[0]


def test_workflow_uses_canonical_grading_prompt() -> None:
    chat = FakeChat([GradingDecision(decision="yes", rationale="Relevant evidence.")])
    evidence = RetrievedEvidence(
        text="The endpoint is /health/live.", source_id="health"
    )

    asyncio.run(
        invoke_crag(
            "Which endpoint reports application liveness?",
            vector_store=FakeVectorStore([evidence]),
            chat=chat,
            web_search=FakeWebSearch(),
        )
    )

    assert chat.structured_prompts == [
        build_grading_prompt(
            "Which endpoint reports application liveness?",
            evidence.text,
        )
    ]


def test_readiness_only_and_unrelated_evidence_are_rejected() -> None:
    async def exercise(text: str) -> dict[str, object]:
        return await invoke_crag(
            "Which endpoint reports application liveness?",
            vector_store=FakeVectorStore(
                [RetrievedEvidence(text=text, source_id="doc-1")]
            ),
            chat=FakeChat(
                [GradingDecision(decision="no", rationale="Insufficient evidence.")]
            ),
            web_search=FakeWebSearch(),
        )

    readiness = asyncio.run(
        exercise("The /health/ready endpoint reports readiness checks.")
    )
    unrelated = asyncio.run(
        exercise("The application closes provider clients during shutdown.")
    )

    assert readiness["grading_decisions"][0].decision == "no"
    assert unrelated["grading_decisions"][0].decision == "no"


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


def test_repeated_relevant_grading_remains_accepted() -> None:
    async def exercise() -> list[str]:
        decisions = []
        for _ in range(5):
            chat = FakeChat(
                [GradingDecision(decision="yes", rationale="Directly relevant")]
            )
            state = await invoke_crag(
                "Which endpoint reports application liveness?",
                vector_store=FakeVectorStore(
                    [
                        RetrievedEvidence(
                            text="The liveness endpoint is /health/live.",
                            source_id="health",
                        )
                    ]
                ),
                chat=chat,
                web_search=FakeWebSearch(),
            )
            decisions.append(state["grading_decisions"][0].decision)
        return decisions

    assert asyncio.run(exercise()) == ["yes"] * 5


def test_provider_timeout_is_not_a_legitimate_rejection() -> None:
    chat = FakeChat(structured_error=ProviderError("grader timed out"))

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Which endpoint reports application liveness?",
            vector_store=FakeVectorStore(
                [
                    RetrievedEvidence(
                        text="The endpoint is /health/live.", source_id="health"
                    )
                ]
            ),
            chat=chat,
            web_search=FakeWebSearch(error=ProviderError("web unavailable")),
        )

    state = asyncio.run(exercise())

    assert state["grading_decisions"] == []
    assert any("Evidence grading failed" in item for item in state["diagnostics"])
    assert state["error"] == "Web search failed: web unavailable"


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


def test_workflow_close_is_idempotent() -> None:
    vector_store = FakeVectorStore()
    workflow = CragWorkflow(
        vector_store=vector_store,
        chat=FakeChat(),
        web_search=FakeWebSearch(),
    )

    async def exercise() -> None:
        await workflow.close()
        await workflow.close()

    asyncio.run(exercise())

    assert vector_store.close_calls == 1


def test_invoke_crag_preserves_caller_owned_resources() -> None:
    vector_store = FakeVectorStore(
        [RetrievedEvidence(text="Evidence", source_id="doc-1")]
    )
    chat = FakeChat([GradingDecision(decision="yes", rationale="Relevant")])

    async def exercise() -> dict[str, object]:
        return await invoke_crag(
            "Question",
            vector_store=vector_store,
            chat=chat,
            web_search=FakeWebSearch(),
        )

    state = asyncio.run(exercise())

    assert state["final_answer"] == "grounded answer"
    assert vector_store.close_calls == 0


@pytest.mark.parametrize("failure", [False, True])
def test_invoke_crag_does_not_close_caller_owned_vector_store(
    failure: bool,
) -> None:
    vector_store = FakeVectorStore(
        error=ProviderError("retrieval failed") if failure else None
    )
    chat = FakeChat(
        [] if failure else [GradingDecision(decision="yes", rationale="Relevant")]
    )

    async def exercise() -> None:
        await invoke_crag(
            "Question",
            vector_store=vector_store,
            chat=chat,
            web_search=FakeWebSearch(),
        )

    asyncio.run(exercise())

    assert vector_store.close_calls == 0


def test_workflow_rejects_invocation_after_close() -> None:
    workflow = CragWorkflow(
        vector_store=FakeVectorStore(),
        chat=FakeChat(),
        web_search=FakeWebSearch(),
    )

    async def exercise() -> None:
        await workflow.close()
        with pytest.raises(RuntimeError, match="closed"):
            await workflow.invoke("Question")

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [False, True])
def test_workflow_closes_explicitly_owned_resources_after_invocation(
    failure: bool,
) -> None:
    vector_store = FakeVectorStore(
        error=ProviderError("retrieval failed") if failure else None
    )
    chat = FakeChat(
        [] if failure else [GradingDecision(decision="yes", rationale="Relevant")]
    )
    web = FakeWebSearch(error=ProviderError("web failed") if failure else None)
    workflow = CragWorkflow(
        vector_store=vector_store,
        chat=chat,
        web_search=web,
        owned_resources=(chat, web),
    )

    async def exercise() -> None:
        await workflow.invoke("Question")
        await workflow.close()

    asyncio.run(exercise())

    assert vector_store.close_calls == 1
    assert chat.close_calls == 1
    assert web.close_calls == 1


def test_workflow_close_failure_keeps_workflow_closed() -> None:
    class ClosingResource:
        def __init__(self, error: Exception | None = None) -> None:
            self.error = error
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.error is not None:
                raise self.error

    first = ClosingResource(RuntimeError("first close failed"))
    second = ClosingResource()
    workflow = CragWorkflow(
        vector_store=FakeVectorStore(),
        chat=FakeChat(),
        web_search=FakeWebSearch(),
        owns_vector_store=False,
        owned_resources=(first, second),
    )

    async def exercise() -> None:
        with pytest.raises(RuntimeError, match="first close failed"):
            await workflow.close()
        with pytest.raises(RuntimeError, match="closed"):
            await workflow.invoke("Question")
        await workflow.close()

    asyncio.run(exercise())

    assert first.close_calls == 1
    assert second.close_calls == 1
