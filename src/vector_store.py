"""
Vector store interface for the FinCrime Intelligence Copilot.

This module wraps interactions with the vector database used for
storing case embeddings.  At this stage it defines a simple API
for adding and querying vectors but does not yet implement any
backend logic.
"""

from typing import List, Tuple


class VectorStore:
    """A dummy vector store implementation."""

    def __init__(self):
        self._vectors: List[Tuple[str, List[float]]] = []

    def add(self, identifier: str, embedding: List[float]):
        """Add an embedding to the store."""
        self._vectors.append((identifier, embedding))

    def query(self, embedding: List[float], top_k: int = 5) -> List[str]:
        """Return the identifiers of the most similar vectors.

        This dummy implementation returns the first `top_k` identifiers.
        """
        return [identifier for identifier, _ in self._vectors[:top_k]]