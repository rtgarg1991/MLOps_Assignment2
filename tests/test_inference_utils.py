import numpy as np

from src.inference import format_probabilities, softmax, top_prediction


def test_softmax_sums_to_one():
    probs = softmax(np.array([1.0, 2.0]))
    assert np.isclose(np.sum(probs), 1.0)


def test_top_prediction_returns_max_entry():
    prob_map = format_probabilities(["cat", "dog"], [0.2, 0.8])
    label, confidence = top_prediction(prob_map)
    assert label == "dog"
    assert confidence == 0.8
