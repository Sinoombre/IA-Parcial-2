"""Modelo L2 y predicciones OOF de desarrollo; nunca carga la prueba reservada."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss
from .features import ALLOWED_PREDICTORS
from .metrics import precision_at_k

NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL = [c for c in ALLOWED_PREDICTORS if c not in NUMERIC]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def make_pipeline(C=1.0):

    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)]
    )
    return Pipeline(
        [
            ("preprocess", prep),
            (
                "model",
                LogisticRegression(
                    C=C, penalty="l2", max_iter=2000, class_weight=None, solver="lbfgs"
                ),
            ),
        ]
    )


def select_C(X, y, customer_id, *, seeds=(1702, 1703, 1704), C_grid=C_GRID):

    rows = []
    for seed in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for C in C_grid:
            ps, ys, ids, fold_scores, losses = [], [], [], [], []
            for train, valid in cv.split(X, y):
                pipe = make_pipeline(C)
                pipe.fit(X.iloc[train], y.iloc[train])
                p = pipe.predict_proba(X.iloc[valid])[:, 1]
                score, _ = precision_at_k(
                    y.iloc[valid].to_numpy(), p, customer_id.iloc[valid].to_numpy()
                )
                fold_scores.append(score)
                losses.append(log_loss(y.iloc[valid], p))
            rows.append(
                {
                    "seed": seed,
                    "C": C,
                    "precision_mean": float(np.nanmean(fold_scores)),
                    "precision_std": float(np.nanstd(fold_scores, ddof=1)),
                    "log_loss_mean": float(np.mean(losses)),
                }
            )

    table = pd.DataFrame(rows)
    grouped = table.groupby("C", as_index=False).agg(
        precision_mean=("precision_mean", "mean"),
        precision_sd=("precision_mean", "std"),
        log_loss_mean=("log_loss_mean", "mean"),
    )
    best = grouped.loc[grouped["precision_mean"].idxmax()]
    threshold = best.precision_mean - (
        0.0 if pd.isna(best.precision_sd) else best.precision_sd / np.sqrt(len(seeds))
    )
    eligible = grouped[grouped.precision_mean >= threshold].sort_values(
        ["precision_mean", "log_loss_mean", "C"], ascending=[False, True, True]
    )
    chosen = float(eligible.iloc[-1]["C"]) if len(eligible) else float(best.C)

    return chosen, table, grouped


def oof_predictions(X, y, customer_id, C, *, seeds=(1702, 1703, 1704)):
    preds = np.zeros(len(X))
    counts = np.zeros(len(X))
    for seed in seeds:
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for train, valid in cv.split(X, y):
            pipe = make_pipeline(C)
            pipe.fit(X.iloc[train], y.iloc[train])
            preds[valid] += pipe.predict_proba(X.iloc[valid])[:, 1]
            counts[valid] += 1

    return preds / counts
