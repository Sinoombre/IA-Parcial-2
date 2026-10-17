"""Decisión aislada de modelo predictivo, CRM y simulador."""

from __future__ import annotations

import math
from typing import Iterable, Mapping


def _sort_key(row: Mapping[str, object]) -> tuple[float, float, str]:
    """Orden contractual: utilidad desc., probabilidad desc., ID asc."""
    return (-float(row["u_act"]), -float(row["p_churn"]), str(row["customer_id"]))


def select_candidates(rows: Iterable[Mapping[str, object]], capacity_fraction: float = 0.15) -> dict:
    """Selecciona como máximo floor(fraction*N) candidatos con utilidad positiva.

    La función no ejecuta campañas y no necesita modelo entrenado. Es determinista
    siempre que las filas tengan los tres campos de ranking y un customer_id estable.
    """
    rows = [dict(row) for row in rows]
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction debe estar en (0, 1]")
    
    n_active = len(rows)
    k = math.floor(capacity_fraction * n_active)
    ranked = sorted(rows, key=_sort_key)
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
