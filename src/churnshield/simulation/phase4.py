"""Motor de simulación de Fase 4; cada escenario vive en su propio módulo."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .crm import SimulatedCRM
from ..decision import select_candidates
from ..utility import add_utilities
from ..scenarios.phase4 import SCENARIOS, SCENARIO_CONFIGS, build_fixture


def _rngs(seed: int):
    
    children = np.random.SeedSequence(seed).spawn(3)
    return tuple(np.random.default_rng(child) for child in children)


def update_r_hat(*, r0: float, m0: float, treated_probabilities: Iterable[float], observed_churn: Iterable[int]) -> float:
    probs = np.asarray(list(treated_probabilities), dtype=float)
    outcomes = np.asarray(list(observed_churn), dtype=int)
    if len(probs) != len(outcomes):
        raise ValueError("E y O deben tener la misma longitud")
    E = float(probs.sum())
    O = float(outcomes.sum())
    if E == 0:
        return float(np.clip(r0, 0.0, 1.0))
    return float(np.clip((m0 * r0 + E - O) / (m0 + E), 0.0, 1.0))


def _write_jsonl(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _risk_segments(rows: list[dict]) -> dict[str, str]:
    """Asigna terciles deterministas sin exponer verdad oculta al agente."""
    ordered = sorted(rows, key=lambda row: (-float(row["p_churn"]), str(row["customer_id"])))
    n = len(ordered)
    cut_high = max(1, int(np.ceil(n / 3)))
    cut_medium = max(cut_high + 1, int(np.ceil(2 * n / 3))) if n > 1 else n
    result = {}
    for index, row in enumerate(ordered):
        result[row["customer_id"]] = "high" if index < cut_high else ("medium" if index < cut_medium else "low")
    return result


def _timestamp(seed: int, cycle: int, sequence: int) -> str:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(days=cycle - 1, seconds=sequence)).isoformat().replace("+00:00", "Z")


def run_scenario(scenario: str, *, seed: int = 101, output_dir: str | Path = "artifacts/phase4") -> dict:
    scenario = scenario.upper()
    if scenario == "S01":
        raise PermissionError("S01 está bloqueado hasta congelar el protocolo y autorizarlo")
    if scenario not in SCENARIOS:
        raise ValueError(f"escenario no soportado: {scenario}; opciones={SCENARIOS}")
    scenario_config = SCENARIO_CONFIGS[scenario]
    df, p_map, q_map, meta = build_fixture(scenario_config, seed)
    cycles = scenario_config.cycles
    r0, r_true, m0, cost = 0.25, 0.35, 10.0, scenario_config.cost
    crm = SimulatedCRM(
        timeout_rate=scenario_config.crm_timeout_rate,
        rejected_rate=scenario_config.crm_rejected_rate,
        error_rate=scenario_config.crm_error_rate,
        timeout_retries=1,
    )
    rng_would, rng_saved, rng_crm = _rngs(seed)
    active = set(df["CustomerID"].astype(str))
    r_hat = r0
    public_decisions, outcomes, learning, summaries, perceptions, truth = [], [], [], [], [], []
    mean_cltv = float(df["CLTV"].mean()) if not np.isnan(float(df["CLTV"].mean())) else 1.0
    for cycle in range(1, cycles + 1):
        active_df = df[df["CustomerID"].astype(str).isin(active)]
        rows = []
        for _, item in active_df.iterrows():
            cid = str(item["CustomerID"])
            p_churn = float(p_map[cid])
            v = float(item["CLTV"]) / mean_cltv if mean_cltv else 1.0
            rows.append({"customer_id": cid, "p_churn": p_churn, "v_cliente": v, "r_hat": r_hat})
        rows = add_utilities(rows, r=r_hat, cost=cost)
        segments = _risk_segments(rows)
        selected = select_candidates(rows, ranking="utility")
        selected_ids = {row["customer_id"] for row in selected["decisions"] if row["selected"]}
        treated_p, treated_obs = [], []
        churned = set()
        for sequence, row in enumerate(rows):
            cid = row["customer_id"]
            timestamp = _timestamp(seed, cycle, sequence)
            segment = segments[cid]
            perceptions.append({"run_id": f"{scenario}-{seed}", "scenario_id": scenario, "seed": seed, "cycle": cycle, "timestamp_utc": timestamp, "customer_id": cid, "p_churn": row["p_churn"], "v_cliente": row["v_cliente"], "segment": segment, "r_hat": r_hat})
            crm_result = None
            if cid in selected_ids:
                crm_result = crm.send(f"{scenario}-{seed}-{cycle}-{cid}", cid, rng_crm)
            would = int(df.loc[df["CustomerID"].astype(str).eq(cid), "Churn Value"].iloc[0]) if cycle == 1 else int(rng_would.random() < q_map[cid])
            accepted = bool(crm_result and crm_result.executed)
            saved = int(bool(accepted and would and rng_saved.random() < r_true))
            observed = int(bool(would and not saved))
            if observed:
                churned.add(cid)
            if accepted:
                treated_p.append(row["p_churn"])
                treated_obs.append(observed)
            public_decisions.append({"run_id": f"{scenario}-{seed}", "scenario_id": scenario, "seed": seed, "cycle": cycle, "timestamp_utc": timestamp, "customer_id": cid, "p_churn": row["p_churn"], "v_cliente": row["v_cliente"], "segment": segment, "r_hat": r_hat, "u_act": row["u_act"], "u_noact": row["u_noact"], "rank": next(x["rank"] for x in selected["decisions"] if x["customer_id"] == cid), "k": selected["k"], "selected": int(cid in selected_ids), "action": "retention_campaign" if cid in selected_ids else "no_action", "campaign_id": crm_result.campaign_id if crm_result else "", "crm_status": crm_result.final_status if crm_result else "not_sent"})
            outcomes.append({"run_id": f"{scenario}-{seed}", "scenario_id": scenario, "seed": seed, "cycle": cycle, "timestamp_utc": timestamp, "customer_id": cid, "campaign_id": crm_result.campaign_id if crm_result else "", "first_attempt_status": crm_result.first_outcome if crm_result else "", "second_attempt_status": crm_result.second_outcome if crm_result else "", "final_status": crm_result.final_status if crm_result else "not_sent", "attempts": crm_result.attempts if crm_result else 0, "retry_used": int(crm_result.retry_used) if crm_result else 0, "executed": int(accepted)})
            truth.append({"run_id": f"{scenario}-{seed}", "scenario_id": scenario, "seed": seed, "cycle": cycle, "timestamp_utc": timestamp, "customer_id": cid, "would_churn": would, "saved": saved, "q_i": q_map[cid], "r_true": r_true, "churned_obs": observed})
        r_next = update_r_hat(r0=r0, m0=m0, treated_probabilities=treated_p, observed_churn=treated_obs)
        learning.append({"run_id": f"{scenario}-{seed}", "scenario_id": scenario, "seed": seed, "cycle": cycle, "r_hat_before": r_hat, "r_hat_after": r_next, "treated_count": len(treated_p), "E": float(sum(treated_p)), "O": int(sum(treated_obs)), "abs_error_to_r_true": abs(r_next - r_true)})
        summaries.append({"run_id": f"{scenario}-{seed}", "scenario_id": scenario, "seed": seed, "cycle": cycle, "n_active_before": len(active), "k": selected["k"], "n_selected": selected["n_selected"], "n_churned": len(churned), "n_active_after": len(active - churned), "r_hat_after": r_next})
        r_hat = r_next
        active -= churned
    out = Path(output_dir) / scenario / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(public_decisions).to_csv(out / "decisions.csv", index=False)
    pd.DataFrame(outcomes).to_csv(out / "outcomes.csv", index=False)
    pd.DataFrame(learning).to_csv(out / "learning_state.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "cycle_summary.csv", index=False)
    pd.DataFrame(truth).to_csv(out / "evaluator_truth.csv", index=False)
    _write_jsonl(out / "perceptions.jsonl", perceptions)
    summary = {**meta, "seed": seed, "cycles": cycles, "r_hat_final": r_hat, "records": len(public_decisions), "output_dir": str(out), "crm_statuses": sorted(set(row["final_status"] for row in outcomes))}
    (out / "phase4_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_many(scenarios: Iterable[str], *, seeds=(101, 102, 103, 104, 105), output_dir: str | Path = "artifacts/phase4") -> list[dict]:
    summaries = []
    for scenario in scenarios:
        for seed in seeds:
            summaries.append(run_scenario(scenario, seed=seed, output_dir=output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(Path(output_dir) / "scenario_summary.csv", index=False)
    return summaries
