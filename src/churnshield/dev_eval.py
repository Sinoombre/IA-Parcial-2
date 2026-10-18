"""Evaluación de desarrollo sobre 15 particiones internas; no toca la reserva."""

from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .data import REPO_ROOT, load_dev
from .features import ALLOWED_PREDICTORS
from .model import oof_predictions, select_C
from .metrics import precision_at_k, recall_at_k, lift_at_k, calibration_metrics
from .stacking import (
    STACKING_META_C,
    STACKING_SEEDS,
    make_stacking_pipeline,
    oof_stacking_predictions,
)
from .utility import add_utilities
from .decision import select_candidates
from .settings import AGENT_CONFIG_PATH, THRESHOLDS_CONFIG_PATH

ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "public"
    / "dev_eval"
    / "stacking"
)


def run_dev_eval(*, output_dir: str | Path = ARTIFACT_DIR):

    df, meta = load_dev()
    X = df[list(ALLOWED_PREDICTORS)].copy()
    y = df["Churn Value"].astype(int)
    ids = df["CustomerID"].astype(str)
    chosen_c, cv_table, cv_summary = select_C(X, y, ids, seeds=STACKING_SEEDS)
    baseline_p = oof_predictions(X, y, ids, chosen_c, seeds=STACKING_SEEDS)
    p = oof_stacking_predictions(X, y, seeds=STACKING_SEEDS)
    mean_cltv = float(df["CLTV"].mean())
    rows = [
        {
            "customer_id": customer_id,
            "p_churn": float(prob),
            "v_cliente": float(value) / mean_cltv if mean_cltv else 1.0,
        }
        for customer_id, prob, value in zip(ids, p, df["CLTV"])
    ]
    util_rows = add_utilities(rows)
    utility_policy = select_candidates(util_rows, ranking="utility")
    probability_policy = select_candidates(util_rows, ranking="probability")
    prob_metrics = {
        "precision": precision_at_k(y, p, ids)[0],
        "recall": recall_at_k(y, p, ids)[0],
        "lift": lift_at_k(y, p, ids)[0],
        **calibration_metrics(y, p),
    }
    baseline_metrics = {
        "precision": precision_at_k(y, baseline_p, ids)[0],
        "recall": recall_at_k(y, baseline_p, ids)[0],
        "lift": lift_at_k(y, baseline_p, ids)[0],
        **calibration_metrics(y, baseline_p),
    }
    y_by_id = dict(zip(ids, y.to_numpy()))

    def selected_precision(result):

        selected = [
            row["customer_id"] for row in result["decisions"] if row["selected"]
        ]
        return (
            float(np.mean([y_by_id[customer_id] for customer_id in selected]))
            if selected
            else float("nan")
        )

    out = {
        "metadata": meta,
        "model_type": "stacking_logistic_lightgbm_xgboost_catboost",
        "baseline_logistic_C": chosen_c,
        "probability_ranking": prob_metrics,
        "baseline_logistic_probability_ranking": baseline_metrics,
        "agent_utility_ranking": {
            "precision": selected_precision(utility_policy),
            "n_selected": int(utility_policy["n_selected"]),
            "k": int(utility_policy["k"]),
        },
        "probability_with_utility_filter": {
            "precision": selected_precision(probability_policy),
            "n_selected": int(probability_policy["n_selected"]),
            "k": int(probability_policy["k"]),
        },
        "spearman_p_churn_v": float(spearmanr(p, df["CLTV"] / mean_cltv).statistic),
        "baseline_logistic_cv_summary": cv_summary.to_dict(orient="records"),
    }
    output_path = Path(output_dir).expanduser().resolve()
    commit_hash = _git_commit()
    working_tree_clean = _git_clean()
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "dev_evaluation.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8"
    )
    cv_table.to_csv(output_path / "baseline_logistic_cv_results.csv", index=False)
    final = make_stacking_pipeline(random_state=1702, include_catboost=True)
    final.fit(X, y)
    import joblib

    model_path = output_path / "model_v1.joblib"
    joblib.dump(final, model_path)

    def sha256(path: Path) -> str:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    card = {
        "model_sha256": sha256(model_path),
        "dataset_sha256": meta["dataset_sha256"],
        "partition_sha256": meta["partition_sha256"],
        "config_sha256": sha256(AGENT_CONFIG_PATH),
        "thresholds_sha256": sha256(THRESHOLDS_CONFIG_PATH),
        "dev_rows": len(df),
        "reserved_rows": meta["reserved_count"],
        "model_type": "stacking_logistic_lightgbm_xgboost_catboost",
        "stacking_meta_logistic_C": STACKING_META_C,
        "baseline_logistic_C": chosen_c,
        "features": list(ALLOWED_PREDICTORS),
        "mean_cltv_train": mean_cltv,
        "reserved_evaluation_executed": False,
    }
    (output_path / "model_card.json").write_text(json.dumps(card, indent=2))
    (output_path / "model_manifest.json").write_text(
        json.dumps(
            {
                **card,
                "model_file": "model_v1.joblib",
                "commit_hash": commit_hash,
                "working_tree_clean": working_tree_clean,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def _git_commit() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_clean() -> bool:
    import subprocess

    return not subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main():
    out = run_dev_eval()
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
