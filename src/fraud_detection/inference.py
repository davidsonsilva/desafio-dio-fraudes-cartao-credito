from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import settings


class FraudPredictor:
    def __init__(self, artifact_path: str | Path = settings.artifact_path):
        self.artifact = joblib.load(artifact_path)

    def predict(self, records: list[dict]) -> list[dict]:
        frame = pd.DataFrame(records)
        missing = set(self.artifact["raw_columns"]).difference(frame.columns)
        if missing:
            raise ValueError(f"Campos ausentes: {sorted(missing)}")
        frame = frame[self.artifact["raw_columns"]]
        features = self.artifact["feature_engineer"].transform(frame)
        scaled = self.artifact["scaler"].transform(features)
        model = self.artifact["model"]
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(scaled)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = -np.asarray(model.decision_function(scaled))
        else:
            scores = np.asarray(model.predict(scaled), dtype=float)
        threshold = self.artifact["threshold"]
        return [{"fraud": bool(score >= threshold), "score": float(score), "threshold": float(threshold), "model": self.artifact["model_name"]} for score in scores]
