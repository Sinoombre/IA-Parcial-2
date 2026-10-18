"""Random Forest comparable con la regresión logística de desarrollo."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from churnshield.features import ALLOWED_PREDICTORS
from churnshield.metrics import precision_at_k, recall_at_k, lift_at_k, calibration_metrics

NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL = [c for c in ALLOWED_PREDICTORS if c not in NUMERIC]
RF_SEEDS = (1702, 1703, 1704)
RF_CONFIGS = {
    "rf_base": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": None},
    "rf_regularized": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": None},
    "rf_balanced": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt", "class_weight": "balanced_subsample"},
}


def make_pipeline(config: dict, *, random_state: int = 1702) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    params = dict(config)
    params["random_state"] = random_state
    params["n_jobs"] = -1
    return Pipeline([("preprocess", prep), ("model", RandomForestClassifier(**params))])


def oof_predictions(X: pd.DataFrame, y: pd.Series, ids: pd.Series, config: dict, *, seeds=RF_SEEDS):
    predictions = np.zeros(len(X), dtype=float)
    counts = np.zeros(len(X), dtype=int)
    fold_rows = []
    for seed in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            pipe = make_pipeline(config, random_state=seed * 100 + fold)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            p = pipe.predict_proba(X.iloc[valid_idx])[:, 1]
            predictions[valid_idx] += p
            counts[valid_idx] += 1
            score, meta = precision_at_k(y.iloc[valid_idx].to_numpy(), p, ids.iloc[valid_idx].to_numpy())
            fold_rows.append({"seed": seed, "fold": fold, "precision_at_15": float(score), "log_loss": float(log_loss(y.iloc[valid_idx], p)), "k": int(meta["k"])})
    assert np.all(counts == len(tuple(seeds)))
    return predictions / counts, pd.DataFrame(fold_rows)


def summarize_predictions(y: pd.Series, p: np.ndarray, ids: pd.Series) -> dict:
    return {
        "precision_at_15": float(precision_at_k(y.to_numpy(), p, ids.to_numpy())[0]),
        "recall_at_15": float(recall_at_k(y.to_numpy(), p, ids.to_numpy())[0]),
        "lift_at_15": float(lift_at_k(y.to_numpy(), p, ids.to_numpy())[0]),
        **calibration_metrics(y.to_numpy(), p),
    }


def compare_random_forest(X: pd.DataFrame, y: pd.Series, ids: pd.Series) -> dict:
    candidates = []
    predictions_by_name = {}
    folds_by_name = {}
    for name, config in RF_CONFIGS.items():
        p, folds = oof_predictions(X, y, ids, config)
        summary = summarize_predictions(y, p, ids)
        fold_mean = float(folds["precision_at_15"].mean())
        fold_sd = float(folds["precision_at_15"].std(ddof=1))
        row = {"model": name, **summary, "fold_precision_mean": fold_mean, "fold_precision_sd": fold_sd, "n_folds": len(folds)}
        candidates.append(row); predictions_by_name[name] = p; folds_by_name[name] = folds
    chosen = sorted(candidates, key=lambda r: (-r["fold_precision_mean"], -r["precision_at_15"], r["log_loss"]))[0]["model"]
    return {"candidates": candidates, "chosen_model": chosen, "chosen_config": RF_CONFIGS[chosen], "folds": {k: v.to_dict(orient="records") for k, v in folds_by_name.items()}}


def save_comparison(result: dict) -> None:
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/random_forest_comparison.json").write_text(json.dumps(result, indent=2, default=float))
