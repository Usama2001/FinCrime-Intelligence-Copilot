"""
Prediction utilities for the FinCrime Intelligence Copilot.

This module wraps model inference into a convenient function.  It
handles model loading and feature preparation.  Currently the
implementation assumes that a trained model has been serialised with
joblib and that the caller provides a dictionary of feature values.
"""

import joblib
import numpy as np
import pandas as pd
from typing import Dict


def load_model(path: str):
    """Load a serialised model from disk."""
    return joblib.load(path)


def prepare_features(feature_dict: Dict[str, float]) -> pd.DataFrame:
    """Convert a dictionary of feature values into a DataFrame.

    Args:
        feature_dict: Mapping from feature name to value.

    Returns:
        A DataFrame with one row of features.
    """
    return pd.DataFrame([feature_dict])


def predict(model, features: pd.DataFrame) -> float:
    """Generate a probability prediction using the model.

    Args:
        model: Fitted classifier.
        features: DataFrame of features.

    Returns:
        Predicted probability for the positive class.
    """
    probas = model.predict_proba(features)
    return float(probas[:, 1][0])