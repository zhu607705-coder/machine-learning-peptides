from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List


def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max_value, max(min_value, value))


def normalize_vector(values: List[float], stats: Dict[str, List[float]]) -> List[float]:
    return [
        (value - stats["mean"][index]) / stats["std"][index]
        for index, value in enumerate(values)
    ]


def denormalize_vector(values: List[float], stats: Dict[str, List[float]]) -> List[float]:
    return [
        (value * stats["std"][index]) + stats["mean"][index]
        for index, value in enumerate(values)
    ]


def forward_normalized(model: Dict[str, Any], normalized_features: List[float]) -> List[float]:
    hidden = []
    for row_index, bias in enumerate(model["weights"]["b1"]):
        total = bias
        for column_index, feature in enumerate(normalized_features):
            total += model["weights"]["W1"][row_index][column_index] * feature
        hidden.append(math.tanh(total))

    outputs = []
    for output_index, bias in enumerate(model["weights"]["b2"]):
        total = bias
        for hidden_index, hidden_value in enumerate(hidden):
            total += model["weights"]["W2"][output_index][hidden_index] * hidden_value
        outputs.append(total)
    return outputs


def predict_targets(model: Dict[str, Any], raw_features: List[float]) -> Dict[str, float]:
    normalized_input = normalize_vector(raw_features, model["inputNormalization"])
    normalized_output = forward_normalized(model, normalized_input)
    purity, yield_value = denormalize_vector(normalized_output, model["targetNormalization"])
    return {
        "purity": clamp(purity, 10.0, 99.5),
        "yield": clamp(yield_value, 5.0, 95.0),
    }


def load_model(model_path: Path | None = None) -> Dict[str, Any]:
    resolved_path = model_path or (Path(__file__).resolve().parents[1] / "artifacts" / "peptide-model.json")
    with resolved_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
