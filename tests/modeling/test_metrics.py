import numpy as np
import pytest

from churnshield.metrics import (
    calibration_metrics,
    lift_at_k,
    precision_at_k,
    recall_at_k,
)


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


@pytest.mark.parametrize(
    ("labels", "probabilities", "fraction", "message"),
    [
        ([1, 0], [0.9], 1.0, "misma longitud"),
        ([1, 0], [0.9, float("nan")], 1.0, "probabilidades finitas"),
        ([1, 2], [0.9, 0.1], 1.0, "etiquetas 0 o 1"),
        ([1, 0], [0.9, 0.1], 1.5, r"\[0, 1\]"),
        ([1, 0], [0.9, 0.1], -0.1, r"\[0, 1\]"),
    ],
)
def test_ranking_metrics_reject_invalid_data(labels, probabilities, fraction, message):
    with pytest.raises(ValueError, match=message):
        precision_at_k(labels, probabilities, fraction=fraction)


def test_ranking_metrics_reject_customer_id_length_mismatch():
    with pytest.raises(ValueError, match="customer_id"):
        precision_at_k([1, 0], [0.9, 0.1], customer_id=["A"])


def test_recall_and_lift_have_hand_calculated_values():
    labels = [1, 0, 1, 0]
    probabilities = [0.9, 0.8, 0.7, 0.1]

    recall, recall_meta = recall_at_k(labels, probabilities, fraction=0.5)
    lift, lift_meta = lift_at_k(labels, probabilities, fraction=0.5)

    assert recall == pytest.approx(0.5)
    assert lift == pytest.approx(1.0)
    assert recall_meta == lift_meta == {"k": 2, "reason": None}


def test_calibration_rejects_empty_or_single_class_labels():
    with pytest.raises(ValueError, match="no pueden estar vacíos"):
        calibration_metrics([], [])
    with pytest.raises(ValueError, match="ambas clases"):
        calibration_metrics([1, 1], [0.8, 0.9])


def test_calibration_error_handles_constant_predictions():
    result = calibration_metrics([0, 0, 1, 1], [0.2, 0.2, 0.2, 0.2])

    assert result["ece"] == pytest.approx(0.3)
