"""Decisión aislada de modelo predictivo, CRM y simulador."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def _sort_key(row: Mapping[str, object], ranking: str) -> tuple[float, float, str]:
    """Orden contractual para una política explícita y reproducible."""
    if ranking == "utility":
        return (-float(row["u_act"]), -float(row["p_churn"]), str(row["customer_id"]))

    if ranking == "probability":
        return (-float(row["p_churn"]), -float(row["u_act"]), str(row["customer_id"]))

    raise ValueError("ranking debe ser 'utility' o 'probability'")


def select_candidates(
    rows: Iterable[Mapping[str, object]],
    capacity_fraction: float = 0.15,
    *,
    ranking: str = "utility",
) -> dict:
    """Selecciona como máximo floor(fraction*N) candidatos con utilidad positiva.

    La función no ejecuta campañas y no necesita modelo entrenado. Es determinista
    siempre que las filas tengan los tres campos de ranking y un customer_id estable.
    """
    rows = [dict(row) for row in rows]
    try:
        capacity_fraction = float(capacity_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "capacity_fraction debe ser un número finito en (0, 1]"
        ) from exc
    if not math.isfinite(capacity_fraction) or not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction debe estar en (0, 1]")
    if ranking not in {"utility", "probability"}:
        raise ValueError("ranking debe ser 'utility' o 'probability'")

    customer_ids = []
    for row in rows:
        try:
            customer_id = str(row["customer_id"])
            probability = float(row["p_churn"])
            utility = float(row["u_act"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "cada candidato requiere customer_id, p_churn y u_act válidos"
            ) from exc
        if not customer_id:
            raise ValueError("customer_id no puede estar vacío")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("p_churn debe ser finito y estar en [0, 1]")
        if not math.isfinite(utility):
            raise ValueError("u_act debe ser finito")
        customer_ids.append(customer_id)
    if len(customer_ids) != len(set(customer_ids)):
        raise ValueError("customer_id debe ser único dentro de los candidatos")

    n_active = len(rows)
    k = math.floor(capacity_fraction * n_active)
    ranked = sorted(rows, key=lambda row: _sort_key(row, ranking))
    candidates = [row for row in ranked if float(row["u_act"]) > 0.0]
    selected = candidates[:k]
    selected_ids = {str(row["customer_id"]) for row in selected}
    decisions = []

    for rank, row in enumerate(ranked, start=1):
        item = dict(row)
        item["rank"] = rank
        item["selected"] = str(row["customer_id"]) in selected_ids
        item["action"] = "retention_campaign" if item["selected"] else "no_action"
        decisions.append(item)
    return {
        "n_active": n_active,
        "k": k,
        "n_selected": len(selected),
        "coverage": (len(selected) / n_active) if n_active else 0.0,
        "decisions": decisions,
    }
