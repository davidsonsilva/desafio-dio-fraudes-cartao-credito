import numpy as np
from sklearn.metrics import recall_score

from fraud_detection.training import dynamic_threshold


def test_dynamic_threshold_reaches_recall_target_when_possible():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.20, 0.60, 0.90])
    threshold = dynamic_threshold(labels, scores, minimum_recall=1.0)
    predictions = scores >= threshold
    assert recall_score(labels, predictions) == 1.0
    assert 0 <= threshold <= 1
