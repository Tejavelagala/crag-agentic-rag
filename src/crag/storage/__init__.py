"""Persistence adapters for document metadata and vectors."""

from crag.storage.registry import DocumentRegistry
from crag.storage.vectorstore import QdrantVectorStoreProvider

__all__ = ["DocumentRegistry", "QdrantVectorStoreProvider"]
