"""
Utility functions for the FinCrime Intelligence Copilot.

This module collects miscellaneous helpers used across the project.  It
is intentionally kept lightweight to avoid circular dependencies.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Create or retrieve a logger with a standard configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger