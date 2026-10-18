"""Evaluación reproducible de trazabilidad (M2) y fallas (M3)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

M2_FIELDS = (
    "run_id",
    "scenario_id",
    "seed",
    "cycle",
    "timestamp_utc",
    "customer_id",
    "p_churn",
    "v_cliente",
    "segment",
    "r_hat",
    "u_act",
    "u_noact",
    "rank",
    "k",
    "selected",
    "action",
    "crm_status",
    "campaign_id",
)
M2_PERCEPTION_FIELDS = (
    "run_id",
    "scenario_id",
    "seed",
    "cycle",
    "timestamp_utc",
    "customer_id",
    "p_churn",
    "v_cliente",
    "segment",
    "r_hat",
)
M2_OUTCOME_FIELDS = (
    "run_id",
    "scenario_id",
    "seed",
    "cycle",
    "timestamp_utc",
    "customer_id",
    "campaign_id",
    "final_status",
)


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_m2(run_dir: str | Path) -> dict:
    """Comprueba reconstrucción 1:1 de decisiones desde los logs públicos."""
    root = Path(run_dir)
    decisions = _rows(root / "decisions.csv")
    perceptions = _jsonl(root / "perceptions.jsonl")
    outcomes = _rows(root / "outcomes.csv")

    def index_rows(rows: list[dict], source: str) -> dict[tuple[str, str, str], dict]:
        indexed = {}
        for row in rows:
            key = (
                row.get("run_id", ""),
                str(row.get("cycle", "")),
                row.get("customer_id", ""),
            )
            if key in indexed:
                raise ValueError(f"clave M2 duplicada en {source}: {key}")
            indexed[key] = row
        return indexed

    decision_rows = index_rows(decisions, "decisions.csv")
    perception_rows = index_rows(perceptions, "perceptions.jsonl")
    outcome_rows = index_rows(outcomes, "outcomes.csv")
    if set(decision_rows) != set(perception_rows) or set(decision_rows) != set(
        outcome_rows
    ):
        raise ValueError("decisions, perceptions y outcomes no tienen claves 1:1")

    reconstructible = 0

    for key, row in decision_rows.items():
        perception = perception_rows[key]
        outcome = outcome_rows[key]
        fields_ok = all(field in row for field in M2_FIELDS)
        fields_ok = fields_ok and all(
            field in perception for field in M2_PERCEPTION_FIELDS
        )
        fields_ok = fields_ok and all(field in outcome for field in M2_OUTCOME_FIELDS)
        values_ok = fields_ok and all(
            row.get(field, "") != "" for field in M2_FIELDS if field != "campaign_id"
        )
        values_ok = values_ok and all(
            (
                float(row[field]) == float(perception[field])
                if field in {"p_churn", "v_cliente", "r_hat"}
                else str(row[field]) == str(perception[field])
            )
            for field in M2_PERCEPTION_FIELDS
        )
        values_ok = values_ok and all(
            str(row["crm_status"] if field == "final_status" else row[field])
            == str(outcome[field])
            for field in M2_OUTCOME_FIELDS
        )
        reconstructible += int(fields_ok and values_ok)

    total = len(decisions)
    return {
        "metric": "M2",
        "total_decisions": total,
        "reconstructible_decisions": reconstructible,
        "score": reconstructible / total if total else 0.0,
        "threshold": 1.0,
        "passes": total > 0 and reconstructible == total,
    }


def evaluate_m3(run_dir: str | Path) -> dict:
    """Comprueba continuidad y registro de la falla esperada del escenario."""
    root = Path(run_dir)
    summary = json.loads((root / "scenario_summary.json").read_text(encoding="utf-8"))
    scenario = summary["scenario"]
    decisions = _rows(root / "decisions.csv")
    outcomes = _rows(root / "outcomes.csv")
    cycles = _rows(root / "cycle_summary.csv")
    statuses = Counter(row["final_status"] for row in outcomes)
    checks = {}
    if scenario == "S04":
        checks["small_population"] = all(
            int(row["k"]) == 0 and int(row["n_selected"]) == 0 for row in cycles
        )

    elif scenario == "S06":
        checks["zero_positive_utility"] = (
            bool(decisions)
            and all(float(row["u_act"]) <= 0 for row in decisions)
            and not any(int(row["selected"]) for row in decisions)
        )

    elif scenario == "S08":
        checks["missing_values"] = summary.get(
            "missing_cell_count", 0
        ) >= 3 and summary.get("imputed_missing_count", 0) == summary.get(
            "missing_cell_count", 0
        )

    elif scenario == "S09":
        checks["unknown_category"] = summary.get("unknown_category_handled") is True

    elif scenario == "S10":
        failure_rows = [
            row
            for row in outcomes
            if row["final_status"] in {"rejected", "error", "crm_error"}
        ]
        checks["failure_statuses_registered"] = bool(failure_rows) and all(
            row["final_status"] and int(row["attempts"]) >= 1 for row in failure_rows
        )
        checks["timeout_retry_registered"] = any(
            int(row["retry_used"]) == 1 for row in outcomes
        )
        checks["continuity"] = len(outcomes) == len(decisions)

    else:
        checks["scenario_completed"] = bool(decisions) and len(outcomes) == len(
            decisions
        )

    passed = sum(checks.values())
    return {
        "metric": "M3",
        "scenario": scenario,
        "checks": checks,
        "passed_cases": passed,
        "expected_cases": len(checks),
        "score": passed / len(checks) if checks else 1.0,
        "threshold": 1.0,
        "passes": bool(checks) and passed == len(checks),
        "crm_statuses": dict(statuses),
    }


def evaluate_batch(
    input_dir: str | Path,
    *,
    expected_runs: Iterable[tuple[str, int]] | None = None,
) -> dict:
    root = Path(input_dir)
    runs = sorted(root.glob("S*/seed_*"))
    if not runs:
        raise ValueError(f"No hay ejecuciones de escenarios en {root}")
    if expected_runs is not None:
        expected = {(scenario.upper(), int(seed)) for scenario, seed in expected_runs}
        actual = set()
        for run in runs:
            try:
                seed = int(run.name.removeprefix("seed_"))
            except ValueError as exc:
                raise ValueError(f"Ruta de ejecución inválida: {run}") from exc
            actual.add((run.parent.name.upper(), seed))
        if actual != expected:
            raise ValueError(
                f"Ejecuciones inesperadas: expected={sorted(expected)} "
                f"actual={sorted(actual)}"
            )
    m2_results = [evaluate_m2(path) for path in runs]
    m3_results = [evaluate_m3(path) for path in runs]
    m2_total = sum(item["total_decisions"] for item in m2_results)
    m2_reconstructed = sum(item["reconstructible_decisions"] for item in m2_results)
    m3_cases = sum(item["expected_cases"] for item in m3_results)
    m3_passed = sum(item["passed_cases"] for item in m3_results)
    s10_statuses = Counter()
    for item in m3_results:
        if item["scenario"] == "S10":
            s10_statuses.update(item["crm_statuses"])
    return {
        "m2": {
            "total_decisions": m2_total,
            "reconstructible_decisions": m2_reconstructed,
            "score": m2_reconstructed / m2_total if m2_total else 0.0,
            "threshold": 1.0,
            "passes": m2_total > 0 and m2_reconstructed == m2_total,
        },
        "m3": {
            "passed_cases": m3_passed,
            "expected_cases": m3_cases,
            "score": m3_passed / m3_cases if m3_cases else 0.0,
            "threshold": 1.0,
            "passes": m3_passed == m3_cases,
            "s10_statuses": dict(s10_statuses),
        },
        "runs": len(runs),
        "run_results": {"m2": m2_results, "m3": m3_results},
    }
