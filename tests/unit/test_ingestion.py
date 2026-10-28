"""Document ingestion and chunking behavior tests."""

import pytest
from pydantic import ValidationError

from crag.ingestion import ingest_document


def test_empty_documents_are_rejected() -> None:
    for text in ("", " \t\n"):
        with pytest.raises(ValueError, match="text"):
            ingest_document(text, document_id="doc-1")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parent_size": 0}, "parent_size"),
        ({"parent_size": -1}, "parent_size"),
        ({"child_size": 0}, "child_size"),
        ({"child_size": 11, "parent_size": 10}, "child_size"),
        ({"overlap": -1}, "overlap"),
        ({"child_size": 5, "overlap": 5}, "overlap"),
    ],
)
def test_invalid_chunking_parameters_are_rejected(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ingest_document("some document text", document_id="doc-1", **kwargs)


def test_long_text_creates_ordered_parent_and_child_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = ingest_document(
        text,
        document_id="doc-1",
        parent_size=10,
        child_size=6,
        overlap=2,
    )

    parents = [chunk for chunk in chunks if chunk.parent_id is None]
    children = [chunk for chunk in chunks if chunk.parent_id is not None]

    assert [parent.text for parent in parents] == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert [child.text for child in children] == [
        "abcdef",
        "efghij",
        "klmnop",
        "opqrst",
        "uvwxyz",
    ]
    parent_ids = {parent.chunk_id for parent in parents}
    assert all(child.parent_id in parent_ids for child in children)


def test_child_chunks_overlap_and_cover_each_parent() -> None:
    chunks = ingest_document(
        "0123456789",
        document_id="doc-1",
        parent_size=10,
        child_size=4,
        overlap=1,
    )
    children = [chunk for chunk in chunks if chunk.parent_id is not None]

    assert [child.text for child in children] == ["0123", "3456", "6789"]
    assert children[0].text[-1] == children[1].text[0]
    assert children[1].text[-1] == children[2].text[0]
    assert (
        "".join(
            child.text if index == 0 else child.text[1:]
            for index, child in enumerate(children)
        )
        == "0123456789"
    )


def test_metadata_is_preserved_without_shared_mutable_defaults() -> None:
    metadata = {"source": "unit-test", "page": 1}
    chunks = ingest_document("short text", document_id="doc-1", metadata=metadata)

    assert all(chunk.metadata == metadata for chunk in chunks)
    assert chunks[0].metadata is not metadata
    assert ingest_document("other text", document_id="doc-2")[0].metadata == {}


def test_short_documents_and_ids_are_repeatable() -> None:
    kwargs = {
        "document_id": "doc-1",
        "metadata": {"source": "test"},
        "parent_size": 100,
        "child_size": 20,
        "overlap": 5,
    }

    first = ingest_document("short text", **kwargs)
    second = ingest_document("short text", **kwargs)
    different_document = ingest_document(
        "short text", **{**kwargs, "document_id": "doc-2"}
    )

    assert first == second
    assert [chunk.chunk_id for chunk in first] != [
        chunk.chunk_id for chunk in different_document
    ]
    assert len(first) == 2
    assert first[1].parent_id == first[0].chunk_id


def test_invalid_metadata_is_rejected_by_document_model() -> None:
    with pytest.raises(ValidationError):
        ingest_document(
            "text",
            document_id="doc-1",
            metadata={"not_json": object()},  # type: ignore[dict-item]
        )
