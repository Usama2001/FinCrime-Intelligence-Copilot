"""
NLP pipeline for the FinCrime Intelligence Copilot.

This module provides basic natural language processing functions to
clean, tokenise and embed narrative text.  When fully implemented it
will support text summarisation and risk tagging.  Currently only
stubs are provided.
"""

from typing import List
import re


def clean_text(text: str) -> str:
    """Lowercase and remove non‑alphanumeric characters from the text."""
    return re.sub(r"[^a-z0-9\s]", "", text.lower())


def extract_keywords(text: str, n: int = 5) -> List[str]:
    """Return a dummy list of keywords from the text.

    This is a placeholder implementation.  In the future you
    might use TF‑IDF or a neural model to extract more meaningful
    keywords.
    """
    tokens = clean_text(text).split()
    return tokens[:n]