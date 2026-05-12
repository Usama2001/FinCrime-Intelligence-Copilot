"""
RAG (Retrieval‑Augmented Generation) pipeline for the FinCrime
Intelligence Copilot.

This module defines a simple interface for combining a vector
database with a language model.  It uses dummy implementations and
does not include the actual LLM or retrieval logic.
"""

from typing import List


class DummyRAGChain:
    """A placeholder RAG chain that returns canned answers."""

    def __init__(self):
        self.documents = {
            "aml_policy": "All transactions above $10,000 must be reported.",
            "case_101": "Customer A received multiple large transfers from high‑risk countries.",
        }

    def run(self, query: str) -> str:
        """Return a simple answer to the query.

        Currently this method does not perform any retrieval or generation
        and simply returns a fixed message.
        """
        return "RAG response functionality not implemented yet."