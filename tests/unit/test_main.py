"""Tests for the Uvicorn application entry point."""

import importlib
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import crag.composition


def test_main_import_does_not_build_workflow(monkeypatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("workflow construction must wait for app startup")

    monkeypatch.setattr(crag.composition, "build_workflow", fail_if_called)
    module = importlib.import_module("crag.main")

    assert isinstance(module.app, FastAPI)


def test_main_lifespan_builds_once_and_closes_workflow(monkeypatch) -> None:
    import crag.main

    calls = SimpleNamespace(builds=0, closes=0)

    class FakeWorkflow:
        async def invoke(self, question: str) -> dict[str, object]:
            return {"final_answer": "answer"}

        async def close(self) -> None:
            calls.closes += 1

    async def fake_build_workflow() -> FakeWorkflow:
        calls.builds += 1
        return FakeWorkflow()

    monkeypatch.setattr(crag.main, "build_workflow", fake_build_workflow)

    with TestClient(crag.main.app) as client:
        response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["checks"] == {"application": True}
    assert calls.builds == 1
    assert calls.closes == 1


def test_main_reports_not_ready_outside_lifespan() -> None:
    import crag.main

    response = TestClient(crag.main.app).get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["workflow_configured"] is False


def test_main_clears_workflow_when_startup_fails(monkeypatch) -> None:
    import crag.main

    async def fail_build_workflow() -> object:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(crag.main, "build_workflow", fail_build_workflow)

    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(crag.main.app):
            pass

    response = TestClient(crag.main.app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"]["workflow_configured"] is False
