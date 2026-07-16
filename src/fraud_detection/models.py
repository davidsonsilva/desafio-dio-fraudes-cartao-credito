from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    supervised: bool = True


class PCAAnomalyDetector(BaseEstimator):
    def __init__(self, n_components: int = 12):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.model_ = PCA(n_components=min(self.n_components, X.shape[1] - 1), random_state=42)
        self.model_.fit(X)
        return self

    def decision_function(self, X):
        reconstructed = self.model_.inverse_transform(self.model_.transform(X))
        return -np.mean((np.asarray(X) - reconstructed) ** 2, axis=1)


class KerasBinaryClassifier(BaseEstimator):
    """Adaptador mínimo para comparar uma rede Keras com estimadores sklearn."""

    def __init__(self, epochs: int = 30, batch_size: int = 1024, random_state: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.random_state)
        self.model_ = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Dense(24, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model_.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
        )
        negatives, positives = np.bincount(np.asarray(y, dtype=int), minlength=2)
        class_weight = {0: 1.0, 1: float(negatives / max(positives, 1))}
        self.model_.fit(
            X,
            y,
            validation_split=0.15,
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=4, restore_best_weights=True)],
            verbose=0,
        )
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        positive = self.model_.predict(X, batch_size=self.batch_size, verbose=0).reshape(-1)
        return np.column_stack([1 - positive, positive])


def model_catalog(random_state: int = 42) -> list[ModelSpec]:
    return [
        ModelSpec("logistic_regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ModelSpec("random_forest", RandomForestClassifier(n_estimators=180, class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)),
        ModelSpec("extra_trees", ExtraTreesClassifier(n_estimators=180, class_weight="balanced", n_jobs=-1, random_state=random_state)),
        ModelSpec("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=150, random_state=random_state)),
        ModelSpec("xgboost", XGBClassifier(n_estimators=240, max_depth=5, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, tree_method="hist", eval_metric="aucpr", random_state=random_state, n_jobs=-1)),
        ModelSpec("tensorflow_keras", KerasBinaryClassifier(random_state=random_state)),
        ModelSpec("isolation_forest", IsolationForest(n_estimators=180, contamination="auto", n_jobs=-1, random_state=random_state), False),
        ModelSpec("local_outlier_factor", LocalOutlierFactor(n_neighbors=35, novelty=True, contamination="auto", n_jobs=-1), False),
        ModelSpec("one_class_svm", OneClassSVM(kernel="rbf", nu=0.01, gamma="scale"), False),
        ModelSpec("pca_reconstruction", PCAAnomalyDetector(), False),
    ]
