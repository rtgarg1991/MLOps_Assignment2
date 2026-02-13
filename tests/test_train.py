from src.train import compute_accuracy


def test_compute_accuracy():
    preds = [0, 1, 1, 0]
    targets = [0, 1, 0, 0]
    accuracy = compute_accuracy(preds, targets)
    assert abs(accuracy - 0.75) < 1e-9


def test_compute_accuracy_empty():
    assert compute_accuracy([], []) == 0.0
