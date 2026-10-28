"""HTTP API tests with an injected fake workflow."""

from fastapi.testclient import TestClient

from crag.api import create_app
from crag.models import GradingDecision, RetrievedEvidence, WebSearchResult
from crag.workflow import CragState


class FakeWorkflow:
    def __init__(self, state: CragState) -> None:
        self.state = state
        self.questions: list[str] = []

    async def invoke(self, question: str) -> CragState:
        self.questions.append(question)
        return self.state


def test_ask_returns_internal_answer_and_safe_sources() -> None:
    workflow = FakeWorkflow(
        {
            "final_answer": "Internal answer",
            "retrieved_evidence": [
                RetrievedEvidence(
                    text="Evidence",
                    source_id="doc-1",
                    metadata={"page": 2, "api_key": "do-not-return"},
                )
            ],
            "grading_decisions": [
                GradingDecision(decision="yes", rationale="Relevant")
            ],
            "web_results": [],
            "fallback_used": False,
            "diagnostics": [],
        }
    )

    response = TestClient(create_app(workflow)).post(
        "/api/v1/ask",
        json={"question": "What is the answer?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Internal answer",
        "sources": [
            {
                "kind": "internal",
                "source_id": "doc-1",
                "title": None,
                "url": None,
                "metadata": {"page": 2, "api_key": "[REDACTED]"},
            }
        ],
        "fallback_used": False,
        "diagnostics": [],
    }


def test_ask_returns_external_fallback_sources_and_safe_diagnostics() -> None:
    workflow = FakeWorkflow(
        {
            "final_answer": "Web answer",
            "retrieved_evidence": [],
            "grading_decisions": [],
            "web_results": [
                WebSearchResult(
                    title="External source",
                    url="https://example.com",
                    snippet="Snippet",
                )
            ],
            "fallback_used": True,
            "diagnostics": ["Internal retrieval failed: secret provider detail"],
        }
    )

    response = TestClient(create_app(workflow)).post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["fallback_used"] is True
    assert body["diagnostics"] == ["internal_retrieval_failed"]
    assert body["sources"][0]["kind"] == "external"
    assert body["sources"][0]["url"] == "https://example.com/"
    assert "secret provider detail" not in response.text


def test_ask_rejects_blank_and_unknown_request_fields() -> None:
    client = TestClient(create_app(FakeWorkflow({})))

    blank = client.post("/api/v1/ask", json={"question": "  "})
    extra = client.post("/api/v1/ask", json={"question": "valid", "api_key": "x"})

    assert blank.status_code == 422
    assert extra.status_code == 422
    assert "api_key" not in extra.text


def test_ask_returns_safe_error_when_workflow_fails() -> None:
    class FailingWorkflow:
        async def invoke(self, question: str) -> CragState:
            raise RuntimeError("secret prompt and provider payload")

    response = TestClient(create_app(FailingWorkflow())).post(
        "/api/v1/ask",
        json={"question": "Question"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The CRAG workflow is temporarily unavailable."
    }
    assert "secret prompt" not in response.text


def test_ask_returns_unavailable_for_workflow_error_state() -> None:
    response = TestClient(
        create_app(
            FakeWorkflow(
                {
                    "error": "Web search failed: provider detail",
                    "fallback_used": True,
                    "diagnostics": ["Web search failed: provider detail"],
                }
            )
        )
    ).post("/api/v1/ask", json={"question": "Question"})

    assert response.status_code == 503
    assert response.json() == {
        "detail": "The CRAG workflow could not produce an answer.",
        "diagnostics": ["web_search_failed"],
    }


def test_health_endpoints_report_only_local_checks() -> None:
    client = TestClient(create_app(FakeWorkflow({})))

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json()["checks"] == {"application": True}
    assert ready.status_code == 200
    assert ready.json()["checks"] == {
        "application": True,
        "workflow_configured": True,
    }
