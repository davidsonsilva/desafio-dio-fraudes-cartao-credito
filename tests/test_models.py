from fraud_detection.models import model_catalog


def test_catalog_has_exactly_ten_distinct_models():
    catalog = model_catalog()
    names = {spec.name for spec in catalog}
    assert len(catalog) == len(names) == 10
    assert {"xgboost", "tensorflow_keras"}.issubset(names)
    assert sum(not spec.supervised for spec in catalog) == 4
