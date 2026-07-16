import itertools

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformação determinística: 30 colunas de origem para mais de 100 features."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        seconds = frame["Time"] % 86400
        angle = 2 * np.pi * seconds / 86400
        frame["hour_sin"] = np.sin(angle)
        frame["hour_cos"] = np.cos(angle)
        frame["day_index"] = frame["Time"] // 86400
        frame["is_night"] = ((seconds < 21600) | (seconds >= 79200)).astype(int)
        amount = frame["Amount"].clip(lower=0)
        frame["amount_log1p"] = np.log1p(amount)
        frame["amount_sqrt"] = np.sqrt(amount)
        frame["amount_cents"] = (amount * 100) % 100
        frame["amount_round_10"] = np.isclose(amount % 10, 0, atol=0.01).astype(int)
        frame["amount_round_100"] = np.isclose(amount % 100, 0, atol=0.01).astype(int)
        frame["amount_time_ratio"] = amount / (frame["Time"] + 1)
        frame["amount_v1_abs"] = amount * frame["V1"].abs()
        frame["amount_v2_abs"] = amount * frame["V2"].abs()
        interaction_columns = [f"V{i}" for i in range(1, 11)]
        for column in interaction_columns:
            frame[f"{column}_sq"] = frame[column] ** 2
            frame[f"{column}_abs"] = frame[column].abs()
        for left, right in itertools.combinations(interaction_columns, 2):
            frame[f"{left}_x_{right}"] = frame[left] * frame[right]
        v_columns = [f"V{i}" for i in range(1, 29)]
        frame["v_mean"] = frame[v_columns].mean(axis=1)
        frame["v_std"] = frame[v_columns].std(axis=1)
        frame["v_abs_max"] = frame[v_columns].abs().max(axis=1)
        frame["v_l2"] = np.sqrt((frame[v_columns] ** 2).sum(axis=1))
        return frame.replace([np.inf, -np.inf], 0).fillna(0)

    def get_feature_names_out(self, input_features=None):
        sample = pd.DataFrame([[0.0] * len(self.feature_names_in_)], columns=self.feature_names_in_)
        return np.asarray(self.transform(sample).columns, dtype=object)
