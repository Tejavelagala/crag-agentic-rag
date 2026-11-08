"""Prepare, verify, and clean up an isolated local CRAG demo."""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient, models

from crag.core.config import Settings
from crag.ingestion import ingest_document
from crag.providers.ollama import OllamaChatProvider, OllamaEmbeddingProvider
from crag.providers.qdrant import QdrantVectorStoreProvider
from crag.rag.workflow import CragWorkflow
from crag.services.composition import DisabledWebSearchProvider

DEFAULT_DATASET_PATH = Path("demo/dataset.json")
DEFAULT_RUNS_PATH = Path("demo/runs")
DEMO_COLLECTION_PREFIX = "crag_demo_"
DEMO_RUN_KEY = "demo_run_id"
DEMO_VERSION_KEY = "demo_dataset_version"


class DemoDocument(BaseModel):
    """A fixed document used by the local demo."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def reject_blank_content(self) -> "DemoDocument":
        if not self.document_id.strip() or not self.text.strip():
            raise ValueError("demo documents require non-blank IDs and text")
        return self


class DemoCase(BaseModel):
    """A smoke question with independently declared expected sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_facts: list[str] = Field(min_length=1)
    expected_source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def reject_blank_values(self) -> "DemoCase":
        values = [
            self.case_id,
            self.question,
            *self.expected_facts,
            *self.expected_source_ids,
        ]
        if any(not value.strip() for value in values):
            raise ValueError("demo cases require non-blank values")
        return self


class DemoDataset(BaseModel):
    """Versioned demo documents and smoke questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(min_length=1)
    documents: list[DemoDocument] = Field(min_length=2, max_length=3)
    cases: list[DemoCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "DemoDataset":
        document_ids = {document.document_id for document in self.documents}
        if len(document_ids) != len(self.documents):
            raise ValueError("demo document_id values must be unique")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("demo case_id values must be unique")
        documents = {
            document.document_id: document.text.lower() for document in self.documents
        }
        for case in self.cases:
            if not set(case.expected_source_ids).issubset(document_ids):
                raise ValueError(
                    f"demo case {case.case_id} references an unknown source"
                )
            source_text = " ".join(
                documents[source_id] for source_id in case.expected_source_ids
            )
            for fact in case.expected_facts:
                if fact.lower() not in source_text:
                    raise ValueError(
                        f"demo fact for case {case.case_id} is absent from "
                        "expected sources"
                    )
        return self


class DemoManifest(BaseModel):
    """Recorded ownership information for one preparation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^demo-[0-9a-f]{16}$")
    collection_name: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    point_ids: list[str] = Field(min_length=1)
    created_at: datetime

    @field_validator("point_ids")
    @classmethod
    def reject_duplicate_point_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("demo manifest point IDs must be unique")
        return value


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> DemoDataset:
    """Load and validate the version-controlled demo corpus."""
    try:
        return DemoDataset.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"Unable to load demo dataset: {path}") from exc


def validate_collection_name(collection_name: str) -> str:
    """Require the dedicated demo namespace."""
    name = collection_name.strip()
    if not name or not name.startswith(DEMO_COLLECTION_PREFIX):
        raise ValueError(
            "Demo collection must be non-blank and begin with 'crag_demo_'"
        )
    return name


def demo_run_id(dataset: DemoDataset, collection_name: str) -> str:
    """Return a stable, non-secret run identifier for this dataset and collection."""
    digest = hashlib.sha256(
        f"{dataset.version}:{collection_name}".encode()
    ).hexdigest()[:16]
    return f"demo-{digest}"


def _chunks(dataset: DemoDataset, run_id: str):
    for document in dataset.documents:
        yield from ingest_document(
            document.text,
            document_id=document.document_id,
            metadata={
                DEMO_RUN_KEY: run_id,
                DEMO_VERSION_KEY: dataset.version,
                "demo": True,
            },
            parent_size=2_000,
            child_size=500,
            overlap=50,
        )


def point_ids(dataset: DemoDataset, run_id: str) -> list[str]:
    """Return deterministic Qdrant point IDs for the demo corpus."""
    return [
        str(uuid5(NAMESPACE_URL, chunk.chunk_id)) for chunk in _chunks(dataset, run_id)
    ]


def _settings_for_demo(settings: Settings, collection_name: str) -> Settings:
    if settings.provider_mode != "local":
        raise ValueError("Demo utility requires CRAG_PROVIDER_MODE=local")
    if not settings.qdrant_url or not settings.qdrant_url.strip():
        raise ValueError("CRAG_QDRANT_URL is required for the demo")
    return settings.model_copy(
        update={"ollama_qdrant_collection_name": collection_name}
    )


def _collection_vectors(info: object) -> tuple[int | None, str | None]:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    distance = getattr(vectors, "distance", None)
    if isinstance(vectors, dict):
        size = vectors.get("size")
        distance = vectors.get("distance")
    return size, str(distance) if distance is not None else None


async def _collection_snapshot(
    client: AsyncQdrantClient,
    collection_name: str,
) -> dict[str, Any]:
    exists = await client.collection_exists(collection_name)
    if not exists:
        return {"exists": False, "point_count": 0}
    info = await client.get_collection(collection_name)
    count = await client.count(collection_name=collection_name, exact=True)
    size, distance = _collection_vectors(info)
    return {
        "exists": True,
        "point_count": count.count,
        "vector_size": size,
        "distance": distance,
    }


async def _owned_points(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    run_id: str,
    dataset_version: str,
) -> list[object]:
    owned: list[object] = []
    offset = None
    while True:
        records, offset = await client.scroll(
            collection_name=collection_name,
            limit=10_000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for record in records:
            payload = getattr(record, "payload", {}) or {}
            metadata = payload.get("metadata", {})
            if (
                isinstance(metadata, dict)
                and metadata.get(DEMO_RUN_KEY) == run_id
                and metadata.get(DEMO_VERSION_KEY) == dataset_version
            ):
                owned.append(record)
        if offset is None:
            break
    return owned


async def prepare(
    dataset: DemoDataset,
    *,
    settings: Settings,
    collection_name: str,
) -> DemoManifest:
    """Prepare demo points after fail-closed collection checks."""
    collection_name = validate_collection_name(collection_name)
    settings = _settings_for_demo(settings, collection_name)
    run_id = demo_run_id(dataset, collection_name)
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )
    embeddings = OllamaEmbeddingProvider(settings)
    chat = OllamaChatProvider(settings)
    vector_store: QdrantVectorStoreProvider | None = None
    try:
        vector = await embeddings.embed_query("CRAG demo dimension probe")
        snapshot = await _collection_snapshot(client, collection_name)
        if snapshot["exists"]:
            if snapshot["point_count"] == 0:
                raise ValueError(
                    "Existing demo collection is empty and ownership is unknown"
                )
            owned = await _owned_points(
                client,
                collection_name,
                run_id=run_id,
                dataset_version=dataset.version,
            )
            expected_ids = set(point_ids(dataset, run_id))
            actual_ids = {str(getattr(point, "id", "")) for point in owned}
            if actual_ids != expected_ids:
                raise ValueError(
                    "Existing demo collection ownership is ambiguous; refusing to write"
                )
            if snapshot["vector_size"] != len(vector):
                raise ValueError("Demo collection vector dimension does not match")
            if snapshot["distance"] != "Cosine":
                raise ValueError("Demo collection must use cosine distance")
        vector_store = QdrantVectorStoreProvider(
            settings,
            embeddings,
            collection_name=collection_name,
            vector_size=len(vector),
            client=client,
        )
        await vector_store.upsert_chunks(list(_chunks(dataset, run_id)))
        manifest = DemoManifest(
            run_id=run_id,
            collection_name=collection_name,
            dataset_version=dataset.version,
            point_ids=point_ids(dataset, run_id),
            created_at=datetime.now(UTC),
        )
        _write_manifest(manifest)
        return manifest
    finally:
        primary_error = sys.exc_info()[1]
        if vector_store is not None:
            close_error = await _close_resources(
                (vector_store, embeddings, chat, client)
            )
        else:
            close_error = await _close_resources((embeddings, chat, client))
        if primary_error is None and close_error is not None:
            raise close_error


async def verify(
    dataset: DemoDataset,
    *,
    settings: Settings,
    collection_name: str,
    question: str | None = None,
) -> dict[str, object]:
    """Verify collection metadata and ownership, optionally running one question."""
    collection_name = validate_collection_name(collection_name)
    settings = _settings_for_demo(settings, collection_name)
    run_id = demo_run_id(dataset, collection_name)
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )
    embeddings = OllamaEmbeddingProvider(settings)
    chat = OllamaChatProvider(settings)
    workflow: CragWorkflow | None = None
    try:
        vector = await embeddings.embed_query("CRAG demo dimension probe")
        snapshot = await _collection_snapshot(client, collection_name)
        if not snapshot["exists"]:
            raise ValueError("Demo collection does not exist")
        owned = await _owned_points(
            client,
            collection_name,
            run_id=run_id,
            dataset_version=dataset.version,
        )
        expected = set(point_ids(dataset, run_id))
        actual = {str(getattr(point, "id", "")) for point in owned}
        if actual != expected:
            raise ValueError("Demo point ownership could not be verified")
        if snapshot["vector_size"] != len(vector):
            raise ValueError("Demo collection vector dimension does not match")
        if snapshot["distance"] != "Cosine":
            raise ValueError("Demo collection must use cosine distance")
        result: dict[str, object] = {
            "collection_metadata_verified": True,
            "collection_ownership_verified": True,
            "embedding_dimensions_compatible": True,
            "demo_points_present": True,
            "run_id": run_id,
            "point_count": snapshot["point_count"],
        }
        if question is not None:
            workflow = CragWorkflow(
                vector_store=QdrantVectorStoreProvider(
                    settings,
                    embeddings,
                    collection_name=collection_name,
                    vector_size=len(vector),
                    client=client,
                ),
                chat=chat,
                web_search=DisabledWebSearchProvider(),
                top_k=1,
                owned_resources=(embeddings, chat),
                web_search_enabled=False,
            )
            state = await workflow.invoke(question)
            result.update(
                {
                    "workflow_smoke_test_completed": True,
                    "grounded_answer_generated": bool(state.get("final_answer")),
                    "fallback_used": state.get("fallback_used", False),
                    "diagnostics": _safe_diagnostics(state),
                    "source_ids": [
                        evidence.source_id
                        for evidence in state.get("retrieved_evidence", [])
                    ],
                }
            )
        return result
    finally:
        primary_error = sys.exc_info()[1]
        if workflow is not None:
            close_error = await _close_resources((workflow, client))
        else:
            close_error = await _close_resources((embeddings, chat, client))
        if primary_error is None and close_error is not None:
            raise close_error


async def cleanup(
    dataset: DemoDataset,
    *,
    settings: Settings,
    collection_name: str,
    run_id: str,
) -> int:
    """Delete only points whose ownership matches the explicit run ID."""
    collection_name = validate_collection_name(collection_name)
    settings = _settings_for_demo(settings, collection_name)
    if not re.fullmatch(r"demo-[0-9a-f]{16}", run_id):
        raise ValueError("Demo run ID has an invalid format")
    expected_run_id = demo_run_id(dataset, collection_name)
    if run_id != expected_run_id:
        raise ValueError("Demo run ID does not match the dataset and collection")
    manifest = _read_manifest(run_id)
    if (
        manifest is None
        or manifest.collection_name != collection_name
        or manifest.dataset_version != dataset.version
        or manifest.run_id != run_id
        or set(manifest.point_ids) != set(point_ids(dataset, run_id))
    ):
        raise ValueError("Recorded demo run ownership could not be verified")
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=(
            settings.qdrant_api_key.get_secret_value()
            if settings.qdrant_api_key is not None
            else None
        ),
    )
    try:
        owned = await _owned_points(
            client,
            collection_name,
            run_id=run_id,
            dataset_version=dataset.version,
        )
        owned_ids = {str(getattr(point, "id", "")) for point in owned}
        expected_ids = set(manifest.point_ids)
        if owned_ids != expected_ids:
            raise ValueError("Demo point ownership could not be verified for cleanup")
        await client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=manifest.point_ids),
            wait=True,
        )
        return len(manifest.point_ids)
    finally:
        primary_error = sys.exc_info()[1]
        close_error = await _close_resources((client,))
        if primary_error is None and close_error is not None:
            raise close_error


def _manifest_path(run_id: str) -> Path:
    return DEFAULT_RUNS_PATH / f"{run_id}.json"


def _write_manifest(manifest: DemoManifest) -> None:
    DEFAULT_RUNS_PATH.mkdir(parents=True, exist_ok=True)
    _manifest_path(manifest.run_id).write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _read_manifest(run_id: str) -> DemoManifest | None:
    path = _manifest_path(run_id)
    if not path.exists():
        return None
    try:
        return DemoManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("Recorded demo run manifest is invalid") from exc


async def _close_resources(resources: tuple[object, ...]) -> Exception | None:
    first_error: Exception | None = None
    for resource in resources:
        close = getattr(resource, "close", None)
        if close is None:
            continue
        try:
            await close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    return first_error


def _safe_diagnostics(state: dict[str, object]) -> list[str]:
    diagnostics: list[str] = []
    for diagnostic in state.get("diagnostics", []):
        if not isinstance(diagnostic, str):
            continue
        if diagnostic.startswith("Internal retrieval failed"):
            diagnostics.append("internal_retrieval_failed")
        elif diagnostic.startswith("Evidence grading failed"):
            diagnostics.append("evidence_grading_failed")
        elif diagnostic.startswith("Web search failed"):
            diagnostics.append("web_search_failed")
        elif diagnostic.startswith("Answer generation failed"):
            diagnostics.append("answer_generation_failed")
    return list(dict.fromkeys(diagnostics))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("prepare", "verify"):
        subparser = subparsers.add_parser(operation)
        subparser.add_argument("--collection", required=True)
        if operation == "verify":
            subparser.add_argument("--question")
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--collection", required=True)
    cleanup_parser.add_argument("--run-id", required=True)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, object]:
    dataset = load_dataset(args.dataset)
    settings = Settings()
    if args.operation == "prepare":
        manifest = await prepare(
            dataset,
            settings=settings,
            collection_name=args.collection,
        )
        return manifest.model_dump(mode="json")
    if args.operation == "verify":
        return await verify(
            dataset,
            settings=settings,
            collection_name=args.collection,
            question=args.question,
        )
    deleted = await cleanup(
        dataset,
        settings=settings,
        collection_name=args.collection,
        run_id=args.run_id,
    )
    return {"deleted_point_count": deleted}


def main() -> None:
    args = _parse_args()
    print(json.dumps(asyncio.run(_run(args)), indent=2))


if __name__ == "__main__":
    main()
