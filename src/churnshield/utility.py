"""Utilidad económica relativa de una acción."""

from __future__ import annotations


def action_utility(p_churn: float, v_cliente: float, *, r: float = 0.25, cost: float = 0.05) -> float:
    return float(p_churn) * float(r) * float(v_cliente) - float(cost)


def add_utilities(rows, *, r: float = 0.25, cost: float = 0.05):
    result = []
    for row in rows:
        item = dict(row)
        item["u_act"] = action_utility(item["p_churn"], item["v_cliente"], r=r, cost=cost)
        item["u_noact"] = 0.0
        result.append(item)
    return result