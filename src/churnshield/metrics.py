"""Métricas predictivas sin usar la partición reservada."""

from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def _top_indices(prob, customer_id=None, fraction=0.15):

    n = len(prob)
    k = math.floor(fraction * n)
    if k == 0:
        return [], k, "k=0"
    ids = np.arange(n) if customer_id is None else np.asarray(customer_id).astype(str)
    order = sorted(range(n), key=lambda i: (-float(prob[i]), str(ids[i])))
    return order[:k], k, None


def precision_at_k(y, prob, customer_id=None, fraction=0.15):

    idx, k, reason = _top_indices(prob, customer_id, fraction)
    if not idx:
        return float("nan"), {"k": k, "reason": reason or "no_selected"}
    return float(np.asarray(y)[idx].mean()), {"k": k, "reason": None}


def recall_at_k(y, prob, customer_id=None, fraction=0.15):

    idx, k, reason = _top_indices(prob, customer_id, fraction)
    total = int(np.asarray(y).sum())
    if not idx or total == 0:
        return float("nan"), {"k": k, "reason": reason or "no_positives"}
    return float(np.asarray(y)[idx].sum() / total), {"k": k, "reason": None}


def lift_at_k(y, prob, customer_id=None, fraction=0.15):

    precision, meta = precision_at_k(y, prob, customer_id, fraction)
    base = float(np.mean(y))
    return (float("nan") if base == 0 else precision / base), meta


def calibration_metrics(y, prob):

    y, prob = np.asarray(y), np.asarray(prob)
    bins = np.quantile(prob, np.linspace(0, 1, 11))
    bins = np.unique(bins)
    ece = 0.0
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
