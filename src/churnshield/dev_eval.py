"""Evaluación de desarrollo sobre 15 particiones internas; no toca la prueba reservada."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data import load_dev
from .features import ALLOWED_PREDICTORS
from .model import select_C, oof_predictions, make_pipeline
from .metrics import precision_at_k, recall_at_k, lift_at_k, calibration_metrics
from .utility import add_utilities
from .decision import select_candidates


def run_dev_eval():

    df, meta = load_dev(); X = df[ALLOWED_PREDICTORS].copy(); y = df["Churn Value"].astype(int); ids = df["CustomerID"].astype(str)
    C, cv_table, cv_summary = select_C(X, y, ids)
    p = oof_predictions(X, y, ids, C)
    mean_cltv = float(df["CLTV"].mean()); rows = [{**r, "v_cliente": r["v_cliente"] / mean_cltv} for r in rows]
    util_rows = add_utilities(rows)
    agent = select_candidates(util_rows, capacity_fraction=0.15, ranking="utility")
    probability_policy = select_candidates(util_rows, capacity_fraction=0.15, ranking="probability")
    prob_metrics = {"precision": precision_at_k(y, p, ids)[0], "recall": recall_at_k(y, p, ids)[0], "lift": lift_at_k(y, p, ids)[0], **calibration_metrics(y, p)}
    y_by_id = dict(zip(rows_id := [r["customer_id"] for r in util_rows], y.to_numpy()))

    def selected_precision(result):

        selected = [r["customer_id"] for r in result["decisions"] if r["selected"]]
        return float(np.mean([y_by_id[x] for x in selected])) if selected else float("nan")
    out = {"metadata": meta, "chosen_C": C, "probability_ranking": prob_metrics, "agent_utility_ranking": {"precision": selected_precision(agent), "n_selected": int(agent["n_selected"]), "k": int(agent["k"])}, "probability_with_utility_filter": {"precision": selected_precision(probability_policy), "n_selected": int(probability_policy["n_selected"]), "k": int(probability_policy["k"])}, "spearman_p_churn_v": float(spearmanr(p, df["CLTV"] / mean_cltv).statistic), "cv_summary": cv_summary.to_dict(orient="records")}
    Path("artifacts").mkdir(exist_ok=True); Path("artifacts/dev_evaluation.json").write_text(json.dumps(out, indent=2, default=float)); cv_table.to_csv("artifacts/cv_results.csv", index=False)
    final = make_pipeline(C); final.fit(X, y); import joblib; joblib.dump(final, "artifacts/model_v1.joblib")
    card = {"dataset_sha256": meta["dataset_sha256"], "dev_rows": len(df), "reserved_rows": meta["reserved_count"], "chosen_C": C, "features": ALLOWED_PREDICTORS, "mean_cltv_train": mean_cltv, "reserved_evaluation_executed": False}
    Path("artifacts/model_card.json").write_text(json.dumps(card, indent=2))
    return out


def main():
    out = run_dev_eval(); print(json.dumps(out, indent=2, default=float))

if __name__ == "__main__": main()
