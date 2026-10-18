"""Evaluación final de la reserva; requiere autorización explícita para ejecutarse."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import load_dev, load_reserved
from .features import ALLOWED_PREDICTORS
from .model import select_C
from .stacking import (
    STACKING_META_C,
    STACKING_SEEDS,
    make_stacking_pipeline,
)
from .metrics import calibration_metrics
from .utility import add_utilities
from .decision import select_candidates
from .protocol import validate_final_authorization
from .settings import THRESHOLDS_CONFIG_PATH, threshold_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "agent.yaml"
THRESHOLD = float(
    threshold_config()["medidas_formales"]["M1_precision_at_15"]["umbral"]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_precision(labels: np.ndarray, *, seeds=range(1, 11)) -> list[float]:
    if len(labels) == 0:
        return []
    rng_values = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sample = rng.choice(labels, size=len(labels), replace=True)
        rng_values.append(float(np.mean(sample)))
    return rng_values


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_final_evaluation(
    *,
    output_dir: str | Path = "artifacts/private/final_evaluation",
    confirm_final: bool = False,
    authorization_file: str | Path | None = None,
) -> dict:
    """Entrena en desarrollo y evalúa una sola vez sobre la reserva.

    El acceso a la reserva ocurre después del guard explícito y nunca durante una
    ejecución normal de desarrollo. El resultado público no contiene etiquetas.
    """
    if not confirm_final:
        raise PermissionError(
            "S01 está bloqueado: use --confirm-final solo con autorización vigente"
        )
    validate_final_authorization(authorization_file, output_dir=output_dir)

    dev, dev_meta = load_dev()
    X_dev = dev[list(ALLOWED_PREDICTORS)].copy()
    y_dev = dev["Churn Value"].astype(int)
    ids_dev = dev["CustomerID"].astype(str)
    chosen_c, cv_table, cv_summary = select_C(
        X_dev, y_dev, ids_dev, seeds=STACKING_SEEDS
    )
    pipeline = make_stacking_pipeline(random_state=1702, include_catboost=True)
    pipeline.fit(X_dev, y_dev)

    reserved = load_reserved(
        authorization_file=authorization_file,
        output_dir=output_dir,
    )
    X_reserved = reserved[list(ALLOWED_PREDICTORS)].copy()
    ids_reserved = reserved["CustomerID"].astype(str)
    probabilities = pipeline.predict_proba(X_reserved)[:, 1]
    mean_cltv = float(dev["CLTV"].mean())
    rows = [
        {
            "customer_id": cid,
            "p_churn": float(prob),
            "v_cliente": float(value) / mean_cltv if mean_cltv else 1.0,
        }
        for cid, prob, value in zip(ids_reserved, probabilities, reserved["CLTV"])
    ]
    policy_rows = add_utilities(rows)
    policy = select_candidates(policy_rows, ranking="utility")
    decisions = policy["decisions"]
    selected_ids = {row["customer_id"] for row in decisions if row["selected"]}
    y_by_id = dict(zip(ids_reserved, reserved["Churn Value"].astype(int)))
    selected_labels = np.asarray([y_by_id[cid] for cid in selected_ids], dtype=int)
    n_churners = int(reserved["Churn Value"].sum())
    precision = float(selected_labels.mean()) if len(selected_labels) else float("nan")
    recall = float(selected_labels.sum() / n_churners) if n_churners else float("nan")
    base_rate = float(reserved["Churn Value"].mean())
    lift = (
        precision / base_rate if base_rate and np.isfinite(precision) else float("nan")
    )
    bootstrap = _bootstrap_precision(selected_labels)
    diagnostics = calibration_metrics(
        reserved["Churn Value"].astype(int).to_numpy(), probabilities
    )

    out = Path(output_dir)
    public_dir = out / "public"
    truth_dir = out / "evaluator_truth"
    public_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    public_rows = []
    perceptions = []
    for row in decisions:
        public = {
            "scenario_id": "S01",
            "customer_id": row["customer_id"],
            "p_churn": row["p_churn"],
            "v_cliente": row["v_cliente"],
            "u_act": row["u_act"],
            "u_noact": row["u_noact"],
            "rank": row["rank"],
            "k": policy["k"],
            "selected": int(row["selected"]),
            "action": row["action"],
            "timestamp_utc": timestamp,
        }
        public_rows.append(public)
        perceptions.append(
            {
                "scenario_id": "S01",
                "customer_id": row["customer_id"],
                "p_churn": row["p_churn"],
                "v_cliente": row["v_cliente"],
                "u_act": row["u_act"],
                "u_noact": row["u_noact"],
                "rank": row["rank"],
                "selected": int(row["selected"]),
                "timestamp_utc": timestamp,
            }
        )

    pd.DataFrame(public_rows).to_csv(
        public_dir / "predictions_reserved.csv", index=False
    )
    pd.DataFrame(public_rows).to_csv(public_dir / "decisions.csv", index=False)
    _write_jsonl(public_dir / "perceptions.jsonl", perceptions)
    pd.DataFrame(
        [{"customer_id": cid, "churn_value": y_by_id[cid]} for cid in ids_reserved]
    ).to_csv(truth_dir / "evaluator_truth.csv", index=False)
    cv_table.to_csv(out / "baseline_logistic_cv_results.csv", index=False)
    joblib.dump(pipeline, out / "model_v1_frozen.joblib")

    metrics = {
        "scenario": "S01",
        "n_reserved": len(reserved),
        "n_selected": policy["n_selected"],
        "k": policy["k"],
        "true_churners_selected": int(selected_labels.sum()),
        "total_churners": n_churners,
        "precision_at_15": precision,
        "recall_at_15": recall,
        "lift_at_15": lift,
        "auc_roc": diagnostics["auc"],
        "brier_score": diagnostics["brier"],
        "log_loss": diagnostics["log_loss"],
        "expected_calibration_error": diagnostics["ece"],
        "m1_threshold": THRESHOLD,
        "m1_passes": bool(np.isfinite(precision) and precision >= THRESHOLD),
        "bootstrap_m1": bootstrap,
        "bootstrap_mean": float(np.mean(bootstrap)) if bootstrap else float("nan"),
        "bootstrap_std": float(np.std(bootstrap, ddof=1))
        if len(bootstrap) > 1
        else float("nan"),
        "model_type": "stacking_logistic_lightgbm_xgboost_catboost",
        "stacking_meta_logistic_C": STACKING_META_C,
        "baseline_logistic_C": chosen_c,
        "dataset_sha256": dev_meta["dataset_sha256"],
        "partition_sha256": dev_meta["partition_sha256"],
        "config_sha256": _sha256(CONFIG_PATH),
        "thresholds_sha256": _sha256(THRESHOLDS_CONFIG_PATH),
        "reserved_access": True,
        "generated_at_utc": timestamp,
    }
    (out / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=float) + "\n"
    )
    (out / "model_manifest.json").write_text(
        json.dumps(
            {
                "model_file": "model_v1_frozen.joblib",
                "model_sha256": _sha256(out / "model_v1_frozen.joblib"),
                "config_file": "config/agent.yaml",
                "config_sha256": _sha256(CONFIG_PATH),
                "thresholds_file": "config/thresholds.yaml",
                "thresholds_sha256": _sha256(THRESHOLDS_CONFIG_PATH),
                "dataset_sha256": dev_meta["dataset_sha256"],
                "partition_sha256": dev_meta["partition_sha256"],
                "dev_rows": len(dev),
                "reserved_rows": len(reserved),
                "model_type": "stacking_logistic_lightgbm_xgboost_catboost",
                "stacking_meta_logistic_C": STACKING_META_C,
                "baseline_logistic_C": chosen_c,
                "features": ALLOWED_PREDICTORS,
                "mean_cltv_train": mean_cltv,
                "reserved_evaluation_executed": True,
            },
            indent=2,
        )
        + "\n"
    )
    (out / "cv_summary.json").write_text(
        json.dumps(cv_summary.to_dict(orient="records"), indent=2) + "\n"
    )
    return metrics
