"""Model training and evaluation."""

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from telco_churn.config import ID_COLUMN, TARGET_COLUMN


def train_and_evaluate(
    dataframe: pd.DataFrame,
    metrics_path: str | Path,
) -> dict[str, float]:
    """Train a CatBoost classifier and persist holdout metrics."""
    target = dataframe[TARGET_COLUMN]
    features = dataframe.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    features = pd.get_dummies(features, drop_first=True, dtype="int8")

    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing values remain in model features: {missing}")
    if target.nunique() != 2:
        raise ValueError("Churn target must contain both classes for evaluation.")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=17,
        stratify=target,
    )
    model = CatBoostClassifier(
        verbose=False,
        random_state=12345,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test).ravel()
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics
