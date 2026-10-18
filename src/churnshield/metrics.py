"""Métricas predictivas sin usar la partición reservada."""

from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def _validated_fraction(fraction):
    try:
        value = float(fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("fraction debe ser un número finito en [0, 1]") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("fraction debe ser un número finito en [0, 1]")
    return value


def _validated_probabilities(prob):
    values = np.asarray(prob, dtype=float)
    if values.ndim != 1:
        raise ValueError("prob debe ser un arreglo unidimensional")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("prob debe contener probabilidades finitas en [0, 1]")
    return values


def _validated_labels(y, expected_length):
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y debe ser un arreglo unidimensional")
    if len(labels) != expected_length:
        raise ValueError("y y prob deben tener la misma longitud")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("y debe contener solo etiquetas 0 o 1")
    return labels.astype(int)


def _top_indices(prob, customer_id=None, fraction=0.15):
    prob = _validated_probabilities(prob)
    fraction = _validated_fraction(fraction)
    n = len(prob)
    k = math.floor(fraction * n)
    if customer_id is None:
        ids = np.arange(n)
    else:
        ids = np.asarray(customer_id)
        if ids.ndim != 1 or len(ids) != n:
            raise ValueError("customer_id debe tener la misma longitud que prob")
        ids = ids.astype(str)
    if k == 0:
        return [], k, "k=0"
    order = sorted(range(n), key=lambda i: (-float(prob[i]), str(ids[i])))
    return order[:k], k, None


def precision_at_k(y, prob, customer_id=None, fraction=0.15):
    prob = _validated_probabilities(prob)
    labels = _validated_labels(y, len(prob))
    idx, k, reason = _top_indices(prob, customer_id, fraction)
    if not idx:
        return float("nan"), {"k": k, "reason": reason or "no_selected"}
    return float(labels[idx].mean()), {"k": k, "reason": None}


def recall_at_k(y, prob, customer_id=None, fraction=0.15):
    prob = _validated_probabilities(prob)
    labels = _validated_labels(y, len(prob))
    idx, k, reason = _top_indices(prob, customer_id, fraction)
    total = int(labels.sum())
    if not idx or total == 0:
        return float("nan"), {"k": k, "reason": reason or "no_positives"}
    return float(labels[idx].sum() / total), {"k": k, "reason": None}


def lift_at_k(y, prob, customer_id=None, fraction=0.15):
    labels = _validated_labels(y, len(_validated_probabilities(prob)))
    precision, meta = precision_at_k(y, prob, customer_id, fraction)
    base = float(np.mean(labels))
    return (float("nan") if base == 0 else precision / base), meta


def calibration_metrics(y, prob):
    prob = _validated_probabilities(prob)
    y = _validated_labels(y, len(prob))
    if not len(y):
        raise ValueError("y y prob no pueden estar vacíos")
    if len(np.unique(y)) < 2:
        raise ValueError("y debe contener ambas clases para calcular AUC y log-loss")
    bins = np.quantile(prob, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    ece = abs(float(y.mean()) - float(prob.mean())) if len(bins) == 1 else 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & ((prob <= hi) if hi == bins[-1] else (prob < hi))
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - prob[mask].mean())
    return {
        "auc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob)),
        "ece": float(ece),
    }
