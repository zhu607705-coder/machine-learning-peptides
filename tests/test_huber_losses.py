from __future__ import annotations

import numpy as np
import torch

from python.losses import build_torch_huber_loss, huber_loss_and_gradient


def test_huber_loss_and_gradient_uses_quadratic_region_for_small_errors() -> None:
    prediction = np.asarray([[0.2, -0.5]], dtype=np.float64)
    target = np.asarray([[0.0, 0.0]], dtype=np.float64)

    loss, gradient = huber_loss_and_gradient(prediction, target, delta=1.0)

    expected_loss = np.asarray([[0.5 * 0.2**2, 0.5 * 0.5**2]], dtype=np.float64)
    expected_gradient = np.asarray([[0.2, -0.5]], dtype=np.float64)
    assert np.allclose(loss, expected_loss)
    assert np.allclose(gradient, expected_gradient)


def test_huber_loss_and_gradient_uses_linear_region_for_large_errors() -> None:
    prediction = np.asarray([[2.0, -3.0]], dtype=np.float64)
    target = np.asarray([[0.0, 0.0]], dtype=np.float64)

    loss, gradient = huber_loss_and_gradient(prediction, target, delta=1.0)

    expected_loss = np.asarray([[1.5, 2.5]], dtype=np.float64)
    expected_gradient = np.asarray([[1.0, -1.0]], dtype=np.float64)
    assert np.allclose(loss, expected_loss)
    assert np.allclose(gradient, expected_gradient)


def test_build_torch_huber_loss_returns_huberloss_with_requested_delta() -> None:
    loss_fn = build_torch_huber_loss(delta=0.75)

    assert isinstance(loss_fn, torch.nn.HuberLoss)
    assert float(loss_fn.delta) == 0.75
