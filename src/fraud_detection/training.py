import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from .config import settings
from .features import FraudFeatureEngineer
from .models import model_catalog


def _scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return -np.asarray(estimator.decision_function(X))
    return np.asarray(estimator.predict(X), dtype=float)


def dynamic_threshold(y_true, scores, minimum_recall: float = 0.80) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    index = eligible[np.argmax(f1[eligible])] if len(eligible) else int(np.argmax(f1))
    return float(thresholds[index])


def train_all(frame: pd.DataFrame, artifact_path: Path = settings.artifact_path) -> dict:
    X = frame.drop(columns="Class")
    y = frame["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, stratify=y, random_state=settings.random_state
    )
    engineer = FraudFeatureEngineer().fit(X_train)
    train_features = engineer.transform(X_train)
    test_features = engineer.transform(X_test)
    scaler = RobustScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    test_scaled = scaler.transform(test_features)
    smote = SMOTE(random_state=settings.random_state)
    train_balanced, y_balanced = smote.fit_resample(train_scaled, y_train)
    normal_train = train_scaled[y_train.to_numpy() == 0]
    # Métodos de kernel/vizinhança não escalam bem para todas as 284 mil linhas.
    if len(normal_train) > 20_000:
        rng = np.random.default_rng(settings.random_state)
        normal_train = normal_train[rng.choice(len(normal_train), 20_000, replace=False)]
    results, fitted = [], {}
    for spec in model_catalog(settings.random_state):
        if spec.name == "xgboost":
            imbalance = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
            spec.estimator.set_params(scale_pos_weight=imbalance)
            fit_X, fit_y = train_scaled, y_train
        elif spec.name == "tensorflow_keras":
            fit_X, fit_y = train_scaled, y_train
        elif spec.supervised:
            fit_X, fit_y = train_balanced, y_balanced
        else:
            fit_X, fit_y = normal_train, None
        spec.estimator.fit(fit_X, fit_y)
        score = _scores(spec.estimator, test_scaled)
        threshold = dynamic_threshold(y_test, score, settings.minimum_recall)
        prediction = (score >= threshold).astype(int)
        metrics = {
            "name": spec.name,
            "supervised": spec.supervised,
            "threshold": threshold,
            "pr_auc": float(average_precision_score(y_test, score)),
            "roc_auc": float(roc_auc_score(y_test, score)),
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
        }
        results.append(metrics)
        fitted[spec.name] = spec.estimator
    winner = max(results, key=lambda item: (item["pr_auc"], item["f1"]))
    artifact = {
        "version": "fraud-detector/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_engineer": engineer,
        "scaler": scaler,
        "model": fitted[winner["name"]],
        "model_name": winner["name"],
        "threshold": winner["threshold"],
        "raw_columns": list(X.columns),
        "feature_count": train_features.shape[1],
        "metrics": results,
        "dataset": {"rows": len(frame), "frauds": int(y.sum()), "fraud_rate": float(y.mean())},
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    settings.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    settings.metrics_path.write_text(json.dumps({k: v for k, v in artifact.items() if k not in {"feature_engineer", "scaler", "model", "raw_columns"}}, indent=2), encoding="utf-8")
    return artifact
