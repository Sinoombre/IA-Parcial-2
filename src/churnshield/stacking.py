"""Cross-fitted stacking model and development-only OOF predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import ALLOWED_PREDICTORS

STACKING_SEEDS = (2006, 2007, 2008)
STACKING_META_C = 10.0
STACKING_INNER_FOLDS = 3
STACKING_OUTER_FOLDS = 5
NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = [column for column in ALLOWED_PREDICTORS if column not in NUMERIC]
    categories = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("cat", categories, categorical),
        ]
    )


def make_stacking_pipeline(
    *,
    random_state: int = 2006,
    include_catboost: bool = False,
) -> StackingClassifier:
    """Create the regularized stack from the configured predictors."""

    def pipeline(model) -> Pipeline:
        return Pipeline([("preprocess", _preprocessor()), ("model", model)])

    estimators = [
        (
            "logistic",
            pipeline(
                LogisticRegression(C=0.1, penalty="l2", max_iter=2000, solver="lbfgs")
            ),
        ),
        (
            "lightgbm",
            pipeline(
                LGBMClassifier(
                    objective="binary",
                    n_estimators=200,
                    learning_rate=0.03,
                    num_leaves=7,
                    max_depth=3,
                    min_child_samples=30,
                    reg_lambda=5,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    random_state=random_state,
                    verbosity=-1,
                    n_jobs=1,
                )
            ),
        ),
        (
            "xgboost",
            pipeline(
                XGBClassifier(
                    n_estimators=250,
                    max_depth=2,
                    learning_rate=0.03,
                    min_child_weight=10,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=10,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                )
            ),
        ),
    ]
    if include_catboost:
        estimators.append(
            (
                "catboost",
                pipeline(
                    CatBoostClassifier(
                        iterations=250,
                        depth=4,
                        learning_rate=0.03,
                        l2_leaf_reg=10,
                        loss_function="Logloss",
                        random_seed=random_state,
                        verbose=False,
                        thread_count=1,
                        allow_writing_files=False,
                    )
                ),
            )
        )

    return StackingClassifier(
        estimators=estimators,
        final_estimator=Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=STACKING_META_C, max_iter=2000)),
            ]
        ),
        cv=StratifiedKFold(
            n_splits=STACKING_INNER_FOLDS,
            shuffle=True,
            random_state=random_state,
        ),
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=False,
    )


def oof_stacking_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seeds: tuple[int, ...] = STACKING_SEEDS,
) -> np.ndarray:
    """Return repeated out-of-fold probabilities using development rows only."""
    predictions = np.zeros(len(X), dtype=float)
    counts = np.zeros(len(X), dtype=int)

    for seed in seeds:
        outer_cv = RepeatedStratifiedKFold(
            n_splits=STACKING_OUTER_FOLDS,
            n_repeats=1,
            random_state=seed,
        )
        for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
            model = make_stacking_pipeline(
                random_state=seed * 10 + fold,
                include_catboost=True,
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions[valid_idx] += model.predict_proba(X.iloc[valid_idx])[:, 1]
            counts[valid_idx] += 1

    if np.any(counts == 0):
        raise RuntimeError("Stacking OOF dejó filas sin predicción")
    return predictions / counts
