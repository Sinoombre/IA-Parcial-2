"""Escenario S03: 50 clientes sintéticos, sin modelo ni simulador."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .decision import select_candidates


def synthetic_rows() -> list[dict]:
    rows = []
    for i in range(50):
        customer_id = f"C{i:03d}"
        if i == 0:
            p_churn, value, utility = 0.90, 1.0, 0.45
        elif i == 1:
            p_churn, value, utility = 0.80, 1.0, 0.30
        elif i == 2:
            p_churn, value, utility = 0.70, 1.0, 0.30
        elif i < 7:
            p_churn, value, utility = 0.60 - 0.02 * i, 1.0, 0.20 - 0.01 * i
        else:
            p_churn, value, utility = 0.10, 1.0, -0.01
        rows.append({
            "customer_id": customer_id,
            "p_churn": p_churn,
            "v_cliente": value,
            "u_act": utility,
            "u_noact": 0.0,
        })
    return rows


def run() -> dict:
    result = select_candidates(synthetic_rows())
    assert result["n_active"] == 50
    assert result["k"] == 7
    assert result["n_selected"] <= 7
    assert result["coverage"] <= 0.15
    assert all(row["u_act"] > 0 for row in result["decisions"] if row["selected"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("S03_OK")
        print(f"n_active={result['n_active']}")
        print(f"k={result['k']}")
        print(f"n_selected={result['n_selected']}")
        print(f"coverage={result['coverage']:.6f}")
        print("selected_ids=" + ",".join(row["customer_id"] for row in result["decisions"] if row["selected"]))


if __name__ == "__main__":
    main()
