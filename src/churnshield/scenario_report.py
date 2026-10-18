"""Reporte ejecutivo de ejecuciones S02--S10.

El reporte resume los artefactos operativos públicos sin leer ni copiar la
verdad privada del evaluador. Produce JSON, CSV y Markdown para facilitar la
demostración de resultados.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from .evaluation import evaluate_batch


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_int(value: object) -> int:
    return int(float(value))


def _as_float(value: object) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _status_text(statuses: dict[str, int]) -> str:
    if not statuses:
        return "sin estados"
    return ", ".join(f"{key}={value}" for key, value in sorted(statuses.items()))


def _run_summary(run_dir: Path, m2: dict, m3: dict) -> dict:
    summary = _read_json(run_dir / "scenario_summary.json")
    decisions = _read_csv(run_dir / "decisions.csv")
    outcomes = _read_csv(run_dir / "outcomes.csv")
    cycles = _read_csv(run_dir / "cycle_summary.csv")
    statuses = Counter(row.get("final_status", "") for row in outcomes)
    selected = sum(_as_int(row.get("selected", 0)) for row in decisions)
    records = len(decisions)
    coverage = selected / records if records else 0.0
    return {
        "scenario": str(summary["scenario"]),
        "seed": _as_int(summary["seed"]),
        "cycles": _as_int(summary["cycles"]),
        "records": records,
        "selected": selected,
        "selection_rate": coverage,
        "mean_k": _mean([_as_float(row["k"]) for row in cycles]),
        "mean_active_before": _mean(
            [_as_float(row["n_active_before"]) for row in cycles]
        ),
        "mean_active_after": _mean(
            [_as_float(row["n_active_after"]) for row in cycles]
        ),
        "r_hat_final": _as_float(summary["r_hat_final"]),
        "crm_statuses": dict(sorted(statuses.items())),
        "unknown_category_handled": bool(
            summary.get("unknown_category_handled", False)
        ),
        "missing_cell_count": _as_int(summary.get("missing_cell_count", 0)),
        "imputed_missing_count": _as_int(summary.get("imputed_missing_count", 0)),
        "m2_score": _as_float(m2["score"]),
        "m3_score": _as_float(m3["score"]),
        "m2_passes": bool(m2["passes"]),
        "m3_passes": bool(m3["passes"]),
    }


def _aggregate_runs(rows: list[dict]) -> list[dict]:
    by_scenario: dict[str, list[dict]] = {}
    for row in rows:
        by_scenario.setdefault(row["scenario"], []).append(row)

    aggregate = []
    for scenario, items in sorted(by_scenario.items()):
        statuses = Counter()
        for item in items:
            statuses.update(item["crm_statuses"])
        aggregate.append(
            {
                "scenario": scenario,
                "runs": len(items),
                "seeds": ",".join(str(item["seed"]) for item in items),
                "mean_records": _mean([item["records"] for item in items]),
                "mean_selected": _mean([item["selected"] for item in items]),
                "mean_selection_rate": _mean(
                    [item["selection_rate"] for item in items]
                ),
                "mean_k": _mean([item["mean_k"] for item in items]),
                "mean_active_before": _mean(
                    [item["mean_active_before"] for item in items]
                ),
                "mean_active_after": _mean(
                    [item["mean_active_after"] for item in items]
                ),
                "mean_r_hat_final": _mean([item["r_hat_final"] for item in items]),
                "crm_statuses": dict(sorted(statuses.items())),
                "m2_score": _mean([item["m2_score"] for item in items]),
                "m3_score": _mean([item["m3_score"] for item in items]),
                "m2_passes": all(item["m2_passes"] for item in items),
                "m3_passes": all(item["m3_passes"] for item in items),
                "unknown_category_handled": any(
                    item["unknown_category_handled"] for item in items
                ),
                "missing_cell_count": sum(item["missing_cell_count"] for item in items),
                "imputed_missing_count": sum(
                    item["imputed_missing_count"] for item in items
                ),
            }
        )
    return aggregate


def build_scenario_report(
    input_dir: str | Path = "artifacts/public/scenarios",
    *,
    evaluation_path: str | Path | None = None,
    expected_runs: Iterable[tuple[str, int]] | None = None,
) -> dict:
    """Construye un resumen por ejecución y un agregado por escenario."""
    root = Path(input_dir)
    run_dirs = sorted(path for path in root.glob("S*/seed_*") if path.is_dir())
    if not run_dirs:
        raise ValueError(f"No hay ejecuciones en {root}")

    if expected_runs is not None:
        evaluation = evaluate_batch(root, expected_runs=expected_runs)
    elif evaluation_path is not None and Path(evaluation_path).exists():
        evaluation = _read_json(Path(evaluation_path))
    else:
        evaluation = evaluate_batch(root)

    m2_by_run = evaluation.get("run_results", {}).get("m2", [])
    m3_by_run = evaluation.get("run_results", {}).get("m3", [])
    if len(m2_by_run) != len(run_dirs) or len(m3_by_run) != len(run_dirs):
        raise ValueError(
            "La evaluación no coincide con el número de ejecuciones encontradas"
        )

    runs = [
        _run_summary(run_dir, m2, m3)
        for run_dir, m2, m3 in zip(run_dirs, m2_by_run, m3_by_run)
    ]
    scenarios = _aggregate_runs(runs)
    return {
        "input_dir": str(root),
        "runs": len(runs),
        "scenarios": len(scenarios),
        "evaluation": {
            "m2_score": evaluation["m2"]["score"],
            "m2_passes": evaluation["m2"]["passes"],
            "m3_score": evaluation["m3"]["score"],
            "m3_passes": evaluation["m3"]["passes"],
            "total_decisions": evaluation["m2"]["total_decisions"],
            "s10_statuses": evaluation["m3"].get("s10_statuses", {}),
        },
        "scenario_summaries": scenarios,
        "run_summaries": runs,
    }


def write_scenario_report(
    report: dict,
    output_dir: str | Path = "artifacts/public/scenario_report",
) -> dict[str, str]:
    """Escribe JSON, CSV y Markdown y devuelve sus rutas."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "scenario_report.json"
    csv_path = out / "scenario_report.csv"
    md_path = out / "scenario_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fields = [
        "scenario",
        "runs",
        "seeds",
        "mean_records",
        "mean_selected",
        "mean_selection_rate",
        "mean_k",
        "mean_active_before",
        "mean_active_after",
        "mean_r_hat_final",
        "m2_score",
        "m3_score",
        "m2_passes",
        "m3_passes",
        "unknown_category_handled",
        "missing_cell_count",
        "imputed_missing_count",
        "crm_statuses",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["scenario_summaries"]:
            item = {key: row.get(key, "") for key in fields}
            item["crm_statuses"] = _status_text(row["crm_statuses"])
            writer.writerow(item)

    evaluation = report["evaluation"]
    lines = [
        "# Resumen ejecutivo de escenarios ChurnShield-Agent",
        "",
        f"- Ejecuciones: **{report['runs']}**",
        f"- Escenarios: **{report['scenarios']}**",
        f"- Decisiones auditadas: **{evaluation['total_decisions']}**",
        f"- M2 trazabilidad: **{evaluation['m2_score']:.3f}** "
        f"({'CUMPLE' if evaluation['m2_passes'] else 'NO CUMPLE'})",
        f"- M3 manejo de fallas: **{evaluation['m3_score']:.3f}** "
        f"({'CUMPLE' if evaluation['m3_passes'] else 'NO CUMPLE'})",
        "",
        "## Lectura rápida",
        "",
        "| Escenario | Runs | Registros promedio | Seleccionados promedio | M2 | M3 | Estados CRM | Interpretación |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in report["scenario_summaries"]:
        if row["scenario"] == "S04":
            interpretation = "Población pequeña; verifica capacidad k=0."
        elif row["scenario"] == "S06":
            interpretation = "Utilidad no positiva; no debe seleccionar."
        elif row["scenario"] == "S08":
            interpretation = (
                f"Faltantes: {row['missing_cell_count']}; "
                f"imputados: {row['imputed_missing_count']}."
            )
        elif row["scenario"] == "S09":
            interpretation = "Categoría desconocida tolerada por el pipeline."
        elif row["scenario"] == "S10":
            interpretation = "CRM adverso; revisar rejected/error/reintentos."
        else:
            interpretation = "Ejecución regular y trazable."
        lines.append(
            f"| {row['scenario']} | {row['runs']} | {row['mean_records']:.1f} | "
            f"{row['mean_selected']:.1f} | {row['m2_score']:.3f} | "
            f"{row['m3_score']:.3f} | {_status_text(row['crm_statuses'])} | "
            f"{interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Estados CRM agregados de S10",
            "",
            "| Estado | Cantidad |",
            "|---|---:|",
        ]
    )
    for status, count in sorted(evaluation["s10_statuses"].items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(
        [
            "",
            "## Comandos de generación",
            "",
            "```bash",
            "export PYTHONPATH=src",
            "python -m churnshield scenarios --all --all-seeds",
            "python -m churnshield evaluate \\",
            "  --expected-scenarios S02 S03 S04 S05 S06 S07 S08 S09 S10 \\",
            "  --expected-seeds 101 102 103 104 105",
            "python -m churnshield scenario-report --all-seeds",
            "```",
            "",
            "El reporte resume artefactos públicos; no copia ni expone la verdad privada del evaluador.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def generate_scenario_report(
    input_dir: str | Path = "artifacts/public/scenarios",
    *,
    evaluation_path: str
    | Path
    | None = "artifacts/public/scenario_evaluation/evaluation.json",
    output_dir: str | Path = "artifacts/public/scenario_report",
    expected_runs: Iterable[tuple[str, int]] | None = None,
) -> dict:
    report = build_scenario_report(
        input_dir, evaluation_path=evaluation_path, expected_runs=expected_runs
    )
    paths = write_scenario_report(report, output_dir)
    return {"report": report, "paths": paths}
