from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def format_probabilities(
    class_names: Iterable[str], probabilities: Iterable[float], precision: int = 6
) -> Dict[str, float]:
    probs = np.asarray(list(probabilities), dtype=np.float64)
    names = list(class_names)
    if len(names) != len(probs):
        raise ValueError("class_names and probabilities length mismatch")
    return {name: round(float(prob), precision) for name, prob in zip(names, probs)}


def top_prediction(prob_map: Dict[str, float]) -> Tuple[str, float]:
    if not prob_map:
        raise ValueError("probability map is empty")
    label = max(prob_map, key=prob_map.get)
    return label, float(prob_map[label])
