from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn


DEFAULT_HUBER_DELTA = 1.0


def build_torch_huber_loss(delta: float = DEFAULT_HUBER_DELTA) -> nn.HuberLoss:
    return nn.HuberLoss(delta=float(delta))


def huber_loss_and_gradient(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    delta: float = DEFAULT_HUBER_DELTA,
) -> Tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    delta = float(delta)
    error = prediction - target
    abs_error = np.abs(error)
    quadratic_mask = abs_error <= delta
    loss = np.where(
        quadratic_mask,
        0.5 * np.square(error),
        delta * (abs_error - 0.5 * delta),
    )
    gradient = np.where(quadratic_mask, error, delta * np.sign(error))
    return loss.astype(np.float64), gradient.astype(np.float64)
