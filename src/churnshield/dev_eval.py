"""Evaluación de desarrollo sobre 15 particiones internas; no toca la reserva."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .data import load_dev
from .features import ALLOWED_PREDICTORS
from .model import select_C, oof_predictions, make_pipeline
from .metrics import precision_at_k, recall_at_k, lift_at_k, calibration_metrics
from .utility import add_utilities
from .decision import select_candidates

ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "dev_eval"


def run_dev_eval():

    df, meta = load_dev()
    X = df[ALLOWED_PREDICTORS].copy()
    y = df["Churn Value"].astype(int)
    ids = df["CustomerID"].astype(str)
    chosen_c, cv_table, cv_summary = select_C(X, y, ids)
    p = oof_predictions(X, y, ids, chosen_c)
    mean_cltv = float(df["CLTV"].mean())
    rows = [{"customer_id": customer_id, "p_churn": float(prob), "v_cliente": float(value) / mean_cltv if mean_cltv else 1.0} for customer_id, prob, value in zip(ids, p, df["CLTV"])]
    util_rows = add_utilities(rows)
    utility_policy = select_candidates(util_rows, ranking="utility")
    probability_policy = select_candidates(util_rows, ranking="probability")
    prob_metrics = {"precision": precision_at_k(y, p, ids)[0], "recall": recall_at_k(y, p, ids)[0], "lift": lift_at_k(y, p, ids)[0], **calibration_metrics(y, p)}
    y_by_id = dict(zip(ids, y.to_numpy()))

    def selected_precision(result):
        
        selected = [row["customer_id"] for row in result["decisions"] if row["selected"]]
        return float(np.mean([y_by_id[customer_id] for customer_id in selected])) if selected else float("nan")

    out = {"metadata": meta, "chosen_C": chosen_c, "probability_ranking": prob_metrics, "agent_utility_ranking": {"precision": selected_precision(utility_policy), "n_selected": int(utility_policy["n_selected"]), "k": int(utility_policy["k"])}, "probability_with_utility_filter": {"precision": selected_precision(probability_policy), "n_selected": int(probability_policy["n_selected"]), "k": int(probability_policy["k"])}, "spearman_p_churn_v": float(spearmanr(p, df["CLTV"] / mean_cltv).statistic), "cv_summary": cv_summary.to_dict(orient="records")}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "dev_evaluation.json").write_text(json.dumps(out, indent=2, default=float))
    cv_table.to_csv(ARTIFACT_DIR / "cv_results.csv", index=False)
    final = make_pipeline(chosen_c)
    final.fit(X, y)
    import joblib
    joblib.dump(final, ARTIFACT_DIR / "model_v1.joblib")
    card = {"dataset_sha256": meta["dataset_sha256"], "dev_rows": len(df), "reserved_rows": meta["reserved_count"], "chosen_C": chosen_c, "features": ALLOWED_PREDICTORS, "mean_cltv_train": mean_cltv, "reserved_evaluation_executed": False}
    (ARTIFACT_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    return out


def main():
    out = run_dev_eval(); print(json.dumps(out, indent=2, default=float))

if __name__ == "__main__": main()
