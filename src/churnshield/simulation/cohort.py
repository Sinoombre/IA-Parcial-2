"""Simulador longitudinal de cohortes de ChurnShield-Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .crm import SimulatedCRM
from .paths import evaluator_output_path
from ..data import load_dev, _read_and_validate
from ..decision import select_candidates
from ..model import oof_predictions, select_C
from ..oracle import oracle_oof_predictions, subset_by_customer_id
from ..utility import add_utilities


def _streams(seed: int) -> dict[str, np.random.Generator]:

    ss = np.random.SeedSequence(int(seed))
    children = ss.spawn(3)
    return {
        name: np.random.default_rng(child)
        for name, child in zip(("would_churn", "saved", "crm"), children)
    }


def _cycle1_truth(df: pd.DataFrame) -> dict[str, int]:
    return dict(zip(df["CustomerID"].astype(str), df["Churn Value"].astype(int)))


def simulate_cohort(
    df: pd.DataFrame,
    p_churn: Mapping[str, float],
    q_i: Mapping[str, float],
    *,
    repetition_seed: int = 101,
    cycles: int = 6,
    r_true: float = 0.35,
    r_action: float = 0.25,
    cost: float = 0.05,
    capacity_fraction: float = 0.15,
    crm: SimulatedCRM | None = None,
    ranking: str = "utility",
) -> tuple[list[dict], list[dict]]:
    """Ejecuta una repetición y devuelve (resumen público, verdad evaluador).

    ``p_churn`` puede ser OOF del modelo de desarrollo. ``q_i`` es siempre una
    entrada separada del evaluador y jamás se inserta en las percepciones.
    """
    if cycles < 1 or not 0 <= r_true <= 1 or not 0 <= r_action <= 1:
        raise ValueError("parámetros inválidos para el simulador")

    required = {"CustomerID", "CLTV", "Churn Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"faltan columnas para simular: {sorted(missing)}")

    ids = df["CustomerID"].astype(str).tolist()
    if set(ids) != set(p_churn) or set(ids) != set(q_i):
        raise ValueError("p_churn y q_i deben cubrir exactamente la cohorte")

    if crm is None:
        crm = SimulatedCRM()

    rng = _streams(repetition_seed)
    mean_cltv = float(df["CLTV"].mean())
    active = {customer_id for customer_id in ids}
    historical = _cycle1_truth(df)
    public_rows: list[dict] = []
    truth_rows: list[dict] = []

    for cycle in range(1, cycles + 1):
        active_df = df[df["CustomerID"].astype(str).isin(active)].copy()
        rows = []
        for _, row in active_df.iterrows():
            customer_id = str(row["CustomerID"])
            rows.append(
                {
                    "customer_id": customer_id,
                    "p_churn": float(p_churn[customer_id]),
                    "v_cliente": float(row["CLTV"]) / mean_cltv if mean_cltv else 1.0,
                }
            )

        rows = add_utilities(rows, r=r_action, cost=cost)
        selected = select_candidates(
            rows, capacity_fraction=capacity_fraction, ranking=ranking
        )
        selected_ids = {
            r["customer_id"] for r in selected["decisions"] if r["selected"]
        }
        churned_this_cycle: set[str] = set()
        for row in rows:
            customer_id = row["customer_id"]

            if cycle == 1:
                would_churn = int(historical[customer_id])

            else:
                would_churn = int(rng["would_churn"].random() < float(q_i[customer_id]))

            crm_result = None
            accepted = False
            if customer_id in selected_ids:
                campaign_id = f"r{repetition_seed}-c{cycle}-{customer_id}"
                crm_result = crm.send(campaign_id, customer_id, rng["crm"])
                accepted = crm_result.executed

            saved = int(
                bool(accepted and would_churn and rng["saved"].random() < r_true)
            )
            churned_obs = int(bool(would_churn and not saved))
            if churned_obs:
                churned_this_cycle.add(customer_id)

            public_rows.append(
                {
                    "repetition_seed": repetition_seed,
                    "cycle": cycle,
                    "customer_id": customer_id,
                    "p_churn": row["p_churn"],
                    "selected": int(customer_id in selected_ids),
                    "action": "retention_campaign"
                    if customer_id in selected_ids
                    else "no_action",
                    "crm_outcome": crm_result.outcome if crm_result else "not_sent",
                    "crm_attempts": crm_result.attempts if crm_result else 0,
                    "churned_obs": churned_obs,
                }
            )

            truth_rows.append(
                {
                    "repetition_seed": repetition_seed,
                    "cycle": cycle,
                    "customer_id": customer_id,
                    "would_churn": would_churn,
                    "saved": saved,
                    "q_i": float(q_i[customer_id]),
                    "r_true": float(r_true),
                    "churned_obs": churned_obs,
                }
            )

        active -= churned_this_cycle
    return public_rows, truth_rows


def development_inputs() -> tuple[
    pd.DataFrame, dict[str, float], dict[str, float], dict
]:
    """Prepara la cohorte de 5.634 con p OOF y q oculto de la población completa."""
    dev, meta = load_dev()
    full, _ = _read_and_validate()
    from ..features import ALLOWED_PREDICTORS

    X, y, ids = (
        dev[ALLOWED_PREDICTORS].copy(),
        dev["Churn Value"].astype(int),
        dev["CustomerID"].astype(str),
    )
    chosen_C, _, _ = select_C(X, y, ids)
    p = oof_predictions(X, y, ids, chosen_C)
    q_full = oracle_oof_predictions(full, seed=1702, target_mean=0.2654)
    q_by_id = subset_by_customer_id(full, full["CustomerID"].astype(str), q_full)
    return (
        dev,
        dict(zip(ids, p)),
        {customer_id: q_by_id[customer_id] for customer_id in ids},
        {
            **meta,
            "chosen_C": chosen_C,
            "oracle_population_mean": float(np.mean(q_full)),
        },
    )


def run_simulation(
    *,
    repetition_seed: int = 101,
    output_dir: str | Path = "artifacts/cohort_simulation",
    evaluator_output_dir: str | Path | None = None,
) -> dict:

    evaluator_root = evaluator_output_path(output_dir, evaluator_output_dir)
    dev, p, q, meta = development_inputs()
    public, truth = simulate_cohort(dev, p, q, repetition_seed=repetition_seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(public).to_csv(out / "cohort_public.csv", index=False)
    truth_out = evaluator_root / "cohort_simulation"
    truth_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(truth).to_csv(truth_out / "evaluator_truth.csv", index=False)

    summary = {
        "repetition_seed": repetition_seed,
        "cycles": 6,
        "dev_rows": len(dev),
        "reserved_access": False,
        "oracle_population_mean": meta["oracle_population_mean"],
        "r_action": 0.25,
        "r_true_evaluator_only": 0.35,
        "truth_file": str(truth_out / "evaluator_truth.csv"),
    }

    (out / "simulation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
