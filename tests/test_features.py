from fraud_detection.data import synthetic_dataset
from fraud_detection.features import FraudFeatureEngineer


def test_feature_engineering_produces_more_than_70_features():
    raw = synthetic_dataset(100).drop(columns="Class")
    transformed = FraudFeatureEngineer().fit_transform(raw)
    assert transformed.shape[1] >= 70
    assert transformed.isna().sum().sum() == 0
    assert list(transformed.index) == list(raw.index)


def test_synthetic_dataset_matches_expected_contract():
    frame = synthetic_dataset(200)
    assert frame.shape == (200, 31)
    assert set(frame["Class"].unique()).issubset({0, 1})
    assert frame["Class"].sum() > 0
