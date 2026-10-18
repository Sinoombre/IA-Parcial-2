"""Evaluación reproducible de trazabilidad (M2) y fallas (M3)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
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


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _record_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("run_id", "")),
        str(row.get("cycle", "")),
        str(row.get("customer_id", "")),
    )


def _same_value(first: dict, second: dict, fields: tuple[str, ...]) -> bool:
    for field in fields:
        if field not in first or field not in second:
            return False
        if field in {"p_churn", "v_cliente", "r_hat"}:
            try:
                if not math.isclose(
                    float(first[field]),
                    float(second[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(first[field]) != str(second[field]):
            return False
    return True


def evaluate_m2(run_dir: str | Path) -> dict:
    """Comprueba reconstrucción 1:1 de decisiones desde los logs públicos."""
    root = Path(run_dir)
    decisions = _rows(root / "decisions.csv")
    perceptions = _jsonl(root / "perceptions.jsonl")
    outcomes = _rows(root / "outcomes.csv")
    decision_counts = Counter(_record_key(row) for row in decisions)
    perceptions_by_key = defaultdict(list)
    outcomes_by_key = defaultdict(list)
    for row in perceptions:
        perceptions_by_key[_record_key(row)].append(row)
    for row in outcomes:
        outcomes_by_key[_record_key(row)].append(row)
    keys_match = len(decisions) == len(perceptions) == len(outcomes) and set(
        decision_counts
    ) == set(perceptions_by_key) == set(outcomes_by_key)
    perception_fields = (
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
    outcome_fields = (
        "run_id",
        "scenario_id",
        "seed",
        "cycle",
        "timestamp_utc",
        "customer_id",
        "campaign_id",
    )
    reconstructible = 0

    for row in decisions:
        key = _record_key(row)
        fields_ok = all(
            field in row and (field == "campaign_id" or row[field] != "")
            for field in M2_FIELDS
        )
        linked = (
            keys_match
            and decision_counts[key] == 1
            and len(perceptions_by_key[key]) == 1
            and len(outcomes_by_key[key]) == 1
        )
        matches = linked and _same_value(
            row, perceptions_by_key[key][0], perception_fields
        )
        matches = matches and _same_value(row, outcomes_by_key[key][0], outcome_fields)
        reconstructible += int(fields_ok and matches)

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
        checks["small_population"] = bool(cycles) and all(
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
            int(row["attempts"]) >= 1 for row in failure_rows
        )
        checks["timeout_retry_registered"] = any(
            int(row["retry_used"]) == 1 for row in outcomes
        )
        checks["continuity"] = bool(decisions) and len(outcomes) == len(decisions)

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


def evaluate_batch(input_dir: str | Path) -> dict:
    root = Path(input_dir)
    runs = sorted(root.glob("S*/seed_*"))
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
            "passes": m3_cases > 0 and m3_passed == m3_cases,
            "s10_statuses": dict(s10_statuses),
        },
        "runs": len(runs),
        "run_results": {"m2": m2_results, "m3": m3_results},
    }
