"""
Model training utilities for the FinCrime Intelligence Copilot.

This module defines functions to train and evaluate various
classification models.  It uses scikit‑learn to build baseline models
such as logistic regression, random forest, XGBoost and LightGBM.  The
implementations are currently placeholders.
"""

from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


def train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Train multiple classifiers and return them.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        A dictionary mapping model names to fitted estimators.
    """
    models = {}

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_train, y_train)
    models["xgboost"] = xgb_model

    # LightGBM
    lgbm = lgb.LGBMClassifier()
    lgbm.fit(X_train, y_train)
    models["lightgbm"] = lgbm

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        model: Fitted model.
        X_test: Test feature matrix.
        y_test: True labels.

    Returns:
        A dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }