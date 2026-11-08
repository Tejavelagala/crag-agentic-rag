"""HTTP API tests with an injected fake workflow."""

from fastapi.testclient import TestClient

from crag import api
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
            "error": None,
            "diagnostics": [],
        }
    )

    response = TestClient(create_app(workflow)).post(
        "/api/v1/ask",
        json={"question": "What is the answer?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Internal answer"
    assert body["sources"] == [
        {
            "kind": "internal",
            "source_id": "doc-1",
            "title": None,
            "url": None,
            "metadata": {"page": 2, "api_key": "[REDACTED]"},
        }
    ]
    assert body["retrieved_evidence"][0]["text"] == "Evidence"
    assert body["grading_decisions"] == [{"decision": "yes", "rationale": "Relevant"}]
    assert body["retrieved_chunk_count"] == 1
    assert body["web_search"]["status"] == "not_attempted"


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


def test_ask_exposes_reported_evidence_stages_and_web_status() -> None:
    workflow = FakeWorkflow(
        {
            "final_answer": "Grounded answer",
            "retrieved_evidence": [
                RetrievedEvidence(
                    text="Relevant evidence",
                    source_id="doc-1",
                    retrieval_score=0.91,
                )
            ],
            "grading_decisions": [
                GradingDecision(decision="yes", rationale="Directly relevant")
            ],
            "web_results": [],
            "fallback_used": False,
            "diagnostics": [],
            "stage_timings_ms": {
                "retrieval": 1.5,
                "grading": 2.5,
                "answer_generation": 3.5,
            },
            "web_search_enabled": False,
            "web_search_attempted": False,
            "web_search_status": "not_attempted",
        }
    )

    body = (
        TestClient(create_app(workflow))
        .post("/api/v1/ask", json={"question": "Question"})
        .json()
    )

    assert body["retrieved_evidence"][0]["retrieval_score"] == 0.91
    assert body["stage_statuses"] == {
        "retrieval": "completed",
        "grading": "completed",
        "routing": "completed",
        "answer": "completed",
    }
    assert body["web_search"] == {
        "enabled": False,
        "attempted": False,
        "status": "not_attempted",
        "source_count": 0,
    }


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
    body = response.json()
    assert body["detail"] == "The CRAG workflow could not produce an answer."
    assert body["diagnostics"] == ["web_search_failed"]
    assert body["answer"] is None
    assert body["web_search"]["status"] == "not_attempted"


def test_read_only_evaluation_reports_are_typed_and_fixed_path(
    monkeypatch, tmp_path
) -> None:
    evaluation_path = tmp_path / "latest.json"
    benchmark_path = tmp_path / "benchmark.json"
    evaluation_path.write_text(
        '{"dataset_version":"1","evaluated_at":"now","provider_mode":"local",'
        '"chat_model":"chat","embedding_model":"embed","collection_name":"c",'
        '"cases":[],"metrics":{"pass_rate":1},"limitations":[]}',
        encoding="utf-8",
    )
    benchmark_path.write_text(
        '{"benchmark_version":"1","evaluated_at":"now","model":"chat",'
        '"production_prompt_version":"semantic-relevance-v2",'
        '"production_prompt_hash":"hash","prompt_variants":{},"limitations":[]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "EVALUATION_REPORT_PATH", evaluation_path)
    monkeypatch.setattr(api, "BENCHMARK_REPORT_PATH", benchmark_path)
    client = TestClient(create_app(FakeWorkflow({})))

    assert client.get("/api/v1/evaluation").json()["metrics"] == {"pass_rate": 1}
    assert client.get("/api/v1/evaluation/benchmark").json()["prompt_variants"] == {}
    assert (
        client.get("/api/v1/evaluation/benchmark").json()["production_prompt_version"]
        == "semantic-relevance-v2"
    )


def test_evaluation_report_errors_are_safe(monkeypatch, tmp_path) -> None:
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(api, "EVALUATION_REPORT_PATH", missing)
    client = TestClient(create_app(FakeWorkflow({})))

    response = client.get("/api/v1/evaluation")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "The requested evaluation report is not available."
    }

    missing.write_text("not-json", encoding="utf-8")
    response = client.get("/api/v1/evaluation")
    assert response.status_code == 500
    assert response.json() == {"detail": "The requested evaluation report is invalid."}


def test_evaluation_and_benchmark_reject_invalid_report_schema(
    monkeypatch, tmp_path
) -> None:
    evaluation_path = tmp_path / "latest.json"
    benchmark_path = tmp_path / "benchmark.json"
    evaluation_path.write_text("{}", encoding="utf-8")
    benchmark_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(api, "EVALUATION_REPORT_PATH", evaluation_path)
    monkeypatch.setattr(api, "BENCHMARK_REPORT_PATH", benchmark_path)
    client = TestClient(create_app(FakeWorkflow({})))

    evaluation = client.get("/api/v1/evaluation")
    benchmark = client.get("/api/v1/evaluation/benchmark")

    assert evaluation.status_code == 500
    assert benchmark.status_code == 500
    assert evaluation.json() == {
        "detail": "The requested evaluation report is invalid."
    }
    assert benchmark.json() == {"detail": "The requested evaluation report is invalid."}


def test_missing_benchmark_report_is_safe(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(api, "BENCHMARK_REPORT_PATH", tmp_path / "missing.json")

    response = TestClient(create_app(FakeWorkflow({}))).get(
        "/api/v1/evaluation/benchmark"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "The requested evaluation report is not available."
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
