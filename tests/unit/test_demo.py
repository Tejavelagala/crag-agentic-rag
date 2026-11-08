"""Offline tests for the isolated demo utility."""

import asyncio
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from crag.config import Settings
from crag.demo import (
    DemoDataset,
    DemoManifest,
    _close_resources,
    cleanup,
    demo_run_id,
    load_dataset,
    point_ids,
    prepare,
    validate_collection_name,
    verify,
)


def test_demo_dataset_loads_and_validates_references() -> None:
    dataset = load_dataset()
    assert len(dataset.documents) == 3
    assert dataset.cases[0].expected_source_ids == ["demo-local-runtime"]


def test_demo_dataset_rejects_duplicate_ids_and_missing_facts() -> None:
    with pytest.raises(ValueError, match="document_id"):
        DemoDataset(
            version="test",
            documents=[
                {"document_id": "doc", "text": "Fact"},
                {"document_id": "doc", "text": "Other"},
            ],
            cases=[
                {
                    "case_id": "case",
                    "question": "What?",
                    "expected_facts": ["Fact"],
                    "expected_source_ids": ["doc"],
                }
            ],
        )
    with pytest.raises(ValueError, match="absent"):
        DemoDataset(
            version="test",
            documents=[
                {"document_id": "doc", "text": "Fact"},
                {"document_id": "doc2", "text": "Other"},
            ],
            cases=[
                {
                    "case_id": "case",
                    "question": "What?",
                    "expected_facts": ["Missing"],
                    "expected_source_ids": ["doc"],
                }
            ],
        )
    with pytest.raises(ValueError, match="non-blank"):
        DemoDataset(
            version="test",
            documents=[
                {"document_id": "doc", "text": "Fact"},
                {"document_id": "doc2", "text": "Other"},
            ],
            cases=[
                {
                    "case_id": "case",
                    "question": "What?",
                    "expected_facts": ["Fact"],
                    "expected_source_ids": [" "],
                }
            ],
        )


@pytest.mark.parametrize(
    "collection",
    ["", "crag_chunks", "crag_evaluation_demo", "crag_integration_demo", "demo"],
)
def test_demo_collection_names_fail_closed(collection: str) -> None:
    with pytest.raises(ValueError, match="crag_demo"):
        validate_collection_name(collection)


def test_demo_point_ids_are_stable() -> None:
    dataset = load_dataset()
    run_id = demo_run_id(dataset, "crag_demo_local")
    assert point_ids(dataset, run_id) == point_ids(dataset, run_id)
    assert run_id.startswith("demo-")


def test_owned_points_scan_all_qdrant_pages() -> None:
    import crag.demo

    client = PaginatedFakeQdrant(
        points=[
            SimpleNamespace(
                id="first",
                payload={
                    "metadata": {"demo_run_id": "run", "demo_dataset_version": "v1"}
                },
            ),
            SimpleNamespace(
                id="second",
                payload={
                    "metadata": {"demo_run_id": "run", "demo_dataset_version": "v1"}
                },
            ),
        ]
    )

    owned = asyncio.run(
        crag.demo._owned_points(
            client,
            "crag_demo_local",
            run_id="run",
            dataset_version="v1",
        )
    )

    assert [point.id for point in owned] == ["first", "second"]


class FakeEmbeddings:
    async def embed_query(self, text: str, *, timeout: float = 30.0) -> list[float]:
        return [1.0, 2.0]

    async def embed_documents(
        self, texts: list[str], *, timeout: float = 30.0
    ) -> list[list[float]]:
        return [[1.0, 2.0] for _ in texts]

    async def close(self) -> None:
        return None


class FakeChat:
    async def generate(self, prompt: str, *, timeout: float = 30.0) -> str:
        return "grounded answer"

    async def generate_structured(
        self, prompt: str, output_type: object, *, timeout: float = 30.0
    ) -> object:
        return output_type(decision="yes", rationale="Relevant")

    async def close(self) -> None:
        return None


class FakeQdrant:
    def __init__(
        self,
        *,
        exists: bool = False,
        points: list[object] | None = None,
        vector_size: int = 2,
        distance: str = "Cosine",
        close_error: Exception | None = None,
        delete_error: Exception | None = None,
    ):
        self.exists = exists
        self.points = points or []
        self.vector_size = vector_size
        self.distance = distance
        self.close_error = close_error
        self.delete_error = delete_error
        self.created = False
        self.deleted: list[str] = []
        self.close_calls = 0

    async def collection_exists(self, name: str) -> bool:
        return self.exists

    async def get_collection(self, name: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(
                        size=self.vector_size,
                        distance=self.distance,
                    )
                )
            )
        )

    async def count(self, collection_name: str, exact: bool) -> object:
        return SimpleNamespace(count=len(self.points))

    async def scroll(self, **kwargs: object) -> tuple[list[object], object]:
        return self.points, None

    async def create_collection(self, **kwargs: object) -> None:
        self.exists = True
        self.created = True

    async def upsert(self, **kwargs: object) -> None:
        self.points = [
            SimpleNamespace(id=point.id, payload=point.payload)
            for point in kwargs["points"]
        ]

    async def delete(self, *, points_selector: object, **kwargs: object) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted = list(points_selector.points)

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def _cleanup_fixture(tmp_path, client: FakeQdrant) -> tuple[DemoDataset, str]:
    dataset = load_dataset()
    run_id = demo_run_id(dataset, "crag_demo_local")
    manifest = DemoManifest(
        run_id=run_id,
        collection_name="crag_demo_local",
        dataset_version=dataset.version,
        point_ids=point_ids(dataset, run_id),
        created_at=datetime.now(UTC),
    )
    tmp_path.joinpath(f"{run_id}.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    client.exists = True
    client.points = [
        SimpleNamespace(
            id=point_id,
            payload={
                "metadata": {
                    "demo_run_id": run_id,
                    "demo_dataset_version": dataset.version,
                }
            },
        )
        for point_id in manifest.point_ids
    ]
    return dataset, run_id


class PaginatedFakeQdrant(FakeQdrant):
    async def scroll(self, **kwargs: object) -> tuple[list[object], object]:
        if kwargs.get("offset") is None:
            return self.points[:1], "next"
        return self.points[1:], None


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        provider_mode="local",
        qdrant_url="http://qdrant.test",
        ollama_qdrant_collection_name="crag_demo_local",
    )


def test_prepare_uses_demo_metadata_and_writes_manifest(monkeypatch, tmp_path) -> None:
    import crag.demo

    client = FakeQdrant()
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        crag.demo, "OllamaEmbeddingProvider", lambda settings: FakeEmbeddings()
    )
    monkeypatch.setattr(crag.demo, "OllamaChatProvider", lambda settings: FakeChat())
    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)

    manifest = asyncio.run(
        prepare(
            load_dataset(),
            settings=_settings(),
            collection_name="crag_demo_local",
        )
    )

    assert client.created is True
    assert len(manifest.point_ids) > 0
    assert (tmp_path / f"{manifest.run_id}.json").exists()
    assert all(
        point.payload["metadata"]["demo_run_id"] == manifest.run_id
        for point in client.points
    )


def test_prepare_rejects_ambiguous_existing_collection(monkeypatch) -> None:
    import crag.demo

    client = FakeQdrant(
        exists=True,
        points=[SimpleNamespace(id="unowned", payload={"metadata": {}})],
    )
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        crag.demo, "OllamaEmbeddingProvider", lambda settings: FakeEmbeddings()
    )
    monkeypatch.setattr(crag.demo, "OllamaChatProvider", lambda settings: FakeChat())

    with pytest.raises(ValueError, match="ownership"):
        asyncio.run(
            prepare(
                load_dataset(),
                settings=_settings(),
                collection_name="crag_demo_local",
            )
        )

    assert client.deleted == []


@pytest.mark.parametrize(
    ("vector_size", "distance", "message"),
    [(3, "Cosine", "dimension"), (2, "Dot", "cosine")],
)
def test_prepare_rejects_incompatible_existing_collection(
    monkeypatch,
    vector_size: int,
    distance: str,
    message: str,
) -> None:
    import crag.demo

    client = FakeQdrant(
        exists=True,
        vector_size=vector_size,
        distance=distance,
    )
    dataset = load_dataset()
    run_id = demo_run_id(dataset, "crag_demo_local")
    client.points = [
        SimpleNamespace(
            id=point_id,
            payload={
                "metadata": {
                    "demo_run_id": run_id,
                    "demo_dataset_version": dataset.version,
                }
            },
        )
        for point_id in point_ids(dataset, run_id)
    ]
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        crag.demo, "OllamaEmbeddingProvider", lambda settings: FakeEmbeddings()
    )
    monkeypatch.setattr(crag.demo, "OllamaChatProvider", lambda settings: FakeChat())

    with pytest.raises(ValueError, match=message):
        asyncio.run(
            prepare(
                dataset,
                settings=_settings(),
                collection_name="crag_demo_local",
            )
        )

    assert client.created is False


def test_verify_rejects_unowned_points(monkeypatch) -> None:
    import crag.demo

    client = FakeQdrant(
        exists=True,
        points=[SimpleNamespace(id="unowned", payload={"metadata": {}})],
    )
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)
    monkeypatch.setattr(
        crag.demo, "OllamaEmbeddingProvider", lambda settings: FakeEmbeddings()
    )
    monkeypatch.setattr(crag.demo, "OllamaChatProvider", lambda settings: FakeChat())

    with pytest.raises(ValueError, match="ownership"):
        asyncio.run(
            verify(
                load_dataset(),
                settings=_settings(),
                collection_name="crag_demo_local",
            )
        )


def test_cleanup_requires_recorded_matching_manifest(monkeypatch, tmp_path) -> None:
    import crag.demo

    dataset = load_dataset()
    run_id = demo_run_id(dataset, "crag_demo_local")
    manifest = DemoManifest(
        run_id=run_id,
        collection_name="crag_demo_local",
        dataset_version=dataset.version,
        point_ids=point_ids(dataset, run_id),
        created_at=datetime.now(UTC),
    )
    tmp_path.joinpath(f"{run_id}.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    client = FakeQdrant(
        exists=True,
        points=[
            SimpleNamespace(
                id=point_id,
                payload={
                    "metadata": {
                        "demo_run_id": run_id,
                        "demo_dataset_version": dataset.version,
                    }
                },
            )
            for point_id in manifest.point_ids
        ],
    )
    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)

    deleted = asyncio.run(
        cleanup(
            dataset,
            settings=_settings(),
            collection_name="crag_demo_local",
            run_id=run_id,
        )
    )

    assert deleted == len(manifest.point_ids)
    assert set(client.deleted) == set(manifest.point_ids)


def test_cleanup_propagates_client_close_failure(monkeypatch, tmp_path) -> None:
    import crag.demo

    client = FakeQdrant(close_error=RuntimeError("client close failed"))
    dataset, run_id = _cleanup_fixture(tmp_path, client)
    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)

    with pytest.raises(RuntimeError, match="client close failed"):
        asyncio.run(
            cleanup(
                dataset,
                settings=_settings(),
                collection_name="crag_demo_local",
                run_id=run_id,
            )
        )

    assert client.deleted
    assert client.close_calls == 1


def test_cleanup_preserves_delete_failure_when_client_close_also_fails(
    monkeypatch,
    tmp_path,
) -> None:
    import crag.demo

    client = FakeQdrant(
        close_error=RuntimeError("client close failed"),
        delete_error=RuntimeError("delete failed"),
    )
    dataset, run_id = _cleanup_fixture(tmp_path, client)
    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)
    monkeypatch.setattr(crag.demo, "AsyncQdrantClient", lambda **kwargs: client)

    with pytest.raises(RuntimeError, match="delete failed"):
        asyncio.run(
            cleanup(
                dataset,
                settings=_settings(),
                collection_name="crag_demo_local",
                run_id=run_id,
            )
        )

    assert client.close_calls == 1


@pytest.mark.parametrize(
    "run_id",
    ["demo-0000000000000000", "../demo-0000000000000000"],
)
def test_cleanup_rejects_altered_or_traversal_run_ids(
    monkeypatch,
    tmp_path,
    run_id: str,
) -> None:
    import crag.demo

    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)
    with pytest.raises(ValueError, match="run ID"):
        asyncio.run(
            cleanup(
                load_dataset(),
                settings=_settings(),
                collection_name="crag_demo_local",
                run_id=run_id,
            )
        )


def test_cleanup_rejects_tampered_manifest_point_ids(monkeypatch, tmp_path) -> None:
    import crag.demo

    dataset = load_dataset()
    run_id = demo_run_id(dataset, "crag_demo_local")
    manifest = DemoManifest(
        run_id=run_id,
        collection_name="crag_demo_local",
        dataset_version=dataset.version,
        point_ids=["tampered-point"],
        created_at=datetime.now(UTC),
    )
    tmp_path.joinpath(f"{run_id}.json").write_text(
        manifest.model_dump_json(), encoding="utf-8"
    )
    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)
    with pytest.raises(ValueError, match="ownership"):
        asyncio.run(
            cleanup(
                dataset,
                settings=_settings(),
                collection_name="crag_demo_local",
                run_id=run_id,
            )
        )


def test_cleanup_rejects_malformed_manifest(monkeypatch, tmp_path) -> None:
    import crag.demo

    dataset = load_dataset()
    run_id = demo_run_id(dataset, "crag_demo_local")
    tmp_path.joinpath(f"{run_id}.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(crag.demo, "DEFAULT_RUNS_PATH", tmp_path)
    with pytest.raises(ValueError, match="manifest is invalid"):
        asyncio.run(
            cleanup(
                dataset,
                settings=_settings(),
                collection_name="crag_demo_local",
                run_id=run_id,
            )
        )


def test_cli_requires_operation_and_collection(monkeypatch) -> None:
    import crag.demo

    monkeypatch.setattr(sys, "argv", ["crag.demo", "prepare"])
    with pytest.raises(SystemExit):
        crag.demo._parse_args()


def test_demo_cleanup_attempts_all_resource_closes() -> None:
    class Resource:
        def __init__(self, fails: bool = False) -> None:
            self.closed = False
            self.fails = fails

        async def close(self) -> None:
            self.closed = True
            if self.fails:
                raise RuntimeError("close failed")

    failing = Resource(fails=True)
    succeeding = Resource()
    error = asyncio.run(_close_resources((failing, succeeding)))

    assert isinstance(error, RuntimeError)
    assert failing.closed is True
    assert succeeding.closed is True
