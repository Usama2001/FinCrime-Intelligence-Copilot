"""
Data preprocessing routines for the FinCrime Intelligence Copilot.

This module defines functions for loading synthetic datasets, cleaning
them and preparing features for modelling.  At present the functions
contain only stub implementations ready for extension.
"""

from typing import Tuple
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the data.
    """
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Perform basic preprocessing on the dataset.

    Splits the DataFrame into features and target and fills missing
    values.  This is a placeholder implementation.

    Args:
        df: Raw DataFrame.

    Returns:
        A tuple of features (X) and target (y).
    """
    # Assume the target column is named 'label'
    y = df["label"] if "label" in df.columns else pd.Series(dtype=int)
    X = df.drop(columns=["label"], errors="ignore")
    X = X.fillna(0)
    return X, y