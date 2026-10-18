import numpy as np
import pytest

from churnshield.metrics import precision_at_k


def test_precision_hand_calculated():
    score, meta = precision_at_k(
        np.array([1, 0, 1, 0, 0, 1, 0]),
        np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        fraction=0.5,
    )
    assert meta["k"] == 3
    assert score == pytest.approx(2 / 3)


def test_k_zero_returns_nan_with_reason():
    score, meta = precision_at_k(np.array([1]), np.array([0.9]), fraction=0.15)
    assert np.isnan(score)
    assert meta["reason"] == "k=0"
