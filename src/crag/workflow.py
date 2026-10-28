"""LangGraph corrective retrieval orchestration."""

import math
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from crag.exceptions import ProviderError
from crag.models import GradingDecision, RetrievedEvidence, WebSearchResult
from crag.providers.base import ChatProvider, VectorStoreProvider, WebSearchProvider

Route = Literal["answer", "web_search", "finish"]


class CragState(TypedDict, total=False):
    """State carried through the corrective retrieval graph."""

    question: str
    retrieved_evidence: list[RetrievedEvidence]
    grading_decisions: list[GradingDecision]
    web_results: list[WebSearchResult]
    final_answer: str | None
    fallback_used: bool
    diagnostics: list[str]
    error: str | None


class CragWorkflow:
    """Build and invoke a corrective retrieval graph with injected providers."""

    def __init__(
        self,
        *,
        vector_store: VectorStoreProvider,
        chat: ChatProvider,
        web_search: WebSearchProvider,
        top_k: int = 5,
        max_web_results: int = 5,
        timeout: float = 30.0,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if max_web_results <= 0:
            raise ValueError("max_web_results must be greater than zero")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite value greater than zero")
        self._vector_store = vector_store
        self._chat = chat
        self._web_search = web_search
        self._top_k = top_k
        self._max_web_results = max_web_results
        self._timeout = timeout
        self._graph = self._build_graph()

    async def invoke(self, question: str) -> CragState:
        """Validate a question and execute the compiled graph asynchronously."""
        if not question.strip():
            raise ValueError("question must not be blank")
        initial_state: CragState = {
            "question": question,
            "retrieved_evidence": [],
            "grading_decisions": [],
            "web_results": [],
            "final_answer": None,
            "fallback_used": False,
            "diagnostics": [],
            "error": None,
        }
        return await self._graph.ainvoke(initial_state)

    def _build_graph(self):
        graph = StateGraph(CragState)
        graph.add_node("retrieve_internal", self._retrieve_internal)
        graph.add_node("grade_evidence", self._grade_evidence)
        graph.add_node("web_search", self._search_web)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("finish", self._finish)

        graph.add_edge(START, "retrieve_internal")
        graph.add_conditional_edges(
            "retrieve_internal",
            self._route_after_retrieval,
            {"grade": "grade_evidence", "web_search": "web_search", "finish": "finish"},
        )
        graph.add_conditional_edges(
            "grade_evidence",
            self._route_after_grading,
            {"answer": "generate_answer", "web_search": "web_search"},
        )
        graph.add_edge("web_search", "generate_answer")
        graph.add_edge("generate_answer", "finish")
        graph.add_edge("finish", END)
        return graph.compile()

    async def _retrieve_internal(self, state: CragState) -> CragState:
        try:
            evidence = await self._vector_store.retrieve(
                state["question"],
                top_k=self._top_k,
                timeout=self._timeout,
            )
            return {"retrieved_evidence": evidence}
        except ProviderError as exc:
            return {
                "retrieved_evidence": [],
                "diagnostics": [f"Internal retrieval failed: {exc}"],
            }

    def _route_after_retrieval(self, state: CragState) -> Route | Literal["grade"]:
        if state.get("diagnostics"):
            return "web_search"
        if state.get("retrieved_evidence"):
            return "grade"
        return "web_search"

    async def _grade_evidence(self, state: CragState) -> CragState:
        decisions: list[GradingDecision] = []
        diagnostics = list(state.get("diagnostics", []))
        for evidence in state["retrieved_evidence"]:
            prompt = (
                "Determine whether the following evidence is relevant to the "
                f"question. Respond with a structured yes/no decision and rationale.\n"
                f"Question: {state['question']}\nEvidence: {evidence.text}"
            )
            try:
                decision = await self._chat.generate_structured(
                    prompt,
                    GradingDecision,
                    timeout=self._timeout,
                )
            except ProviderError as exc:
                diagnostics.append(f"Evidence grading failed: {exc}")
                return {"grading_decisions": [], "diagnostics": diagnostics}
            decisions.append(decision)
        return {"grading_decisions": decisions, "diagnostics": diagnostics}

    def _route_after_grading(self, state: CragState) -> Route:
        decisions = state.get("grading_decisions", [])
        if len(decisions) != len(state.get("retrieved_evidence", [])):
            return "web_search"
        if any(decision.decision == "yes" for decision in decisions):
            return "answer"
        return "web_search"

    async def _search_web(self, state: CragState) -> CragState:
        try:
            results = await self._web_search.search(
                state["question"],
                max_results=self._max_web_results,
                timeout=self._timeout,
            )
            return {
                "web_results": results,
                "fallback_used": True,
            }
        except ProviderError as exc:
            return {
                "web_results": [],
                "fallback_used": True,
                "error": f"Web search failed: {exc}",
                "diagnostics": [f"Web search failed: {exc}"],
            }

    async def _generate_answer(self, state: CragState) -> CragState:
        if state.get("error"):
            return {"final_answer": None}
        sources = self._format_sources(state)
        if not sources:
            return {
                "final_answer": None,
                "error": "No evidence was available to ground an answer",
            }
        prompt = (
            "Answer the question using only the supplied evidence. Distinguish "
            "internal evidence from external web results. Do not invent citations "
            "or facts, and say when the evidence is insufficient.\n"
            f"Question: {state['question']}\n{sources}"
        )
        try:
            answer = await self._chat.generate(prompt, timeout=self._timeout)
            return {"final_answer": answer}
        except ProviderError as exc:
            return {
                "final_answer": None,
                "error": f"Answer generation failed: {exc}",
                "diagnostics": [f"Answer generation failed: {exc}"],
            }

    def _finish(self, state: CragState) -> CragState:
        return state

    @staticmethod
    def _format_sources(state: CragState) -> str:
        parts: list[str] = []
        decisions = state.get("grading_decisions", [])
        for index, evidence in enumerate(state.get("retrieved_evidence", [])):
            if index < len(decisions) and decisions[index].decision == "yes":
                parts.append(f"[Internal source {index + 1}]\n{evidence.text}")
        for index, result in enumerate(state.get("web_results", [])):
            parts.append(
                f"[External web source {index + 1}]\n"
                f"Title: {result.title}\nURL: {result.url}\n{result.snippet}"
            )
        return "\n\n".join(parts)


def build_crag_workflow(
    *,
    vector_store: VectorStoreProvider,
    chat: ChatProvider,
    web_search: WebSearchProvider,
    top_k: int = 5,
    max_web_results: int = 5,
    timeout: float = 30.0,
):
    """Return a compiled corrective retrieval graph with injected providers."""
    return CragWorkflow(
        vector_store=vector_store,
        chat=chat,
        web_search=web_search,
        top_k=top_k,
        max_web_results=max_web_results,
        timeout=timeout,
    )


async def invoke_crag(
    question: str,
    *,
    vector_store: VectorStoreProvider,
    chat: ChatProvider,
    web_search: WebSearchProvider,
    top_k: int = 5,
    max_web_results: int = 5,
    timeout: float = 30.0,
) -> CragState:
    """Execute the corrective retrieval graph asynchronously."""
    workflow = CragWorkflow(
        vector_store=vector_store,
        chat=chat,
        web_search=web_search,
        top_k=top_k,
        max_web_results=max_web_results,
        timeout=timeout,
    )
    return await workflow.invoke(question)
