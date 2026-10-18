"""Compare regularized stacking candidates using development data only."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = str(REPO_ROOT / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from churnshield.metrics import precision_at_k
from churnshield.features import ALLOWED_PREDICTORS
from churnshield.model import make_pipeline
from churnshield.stacking import (
    STACKING_META_C,
    STACKING_SEEDS,
    make_stacking_pipeline,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "public" / "experiments" / "stacking"
SEEDS = STACKING_SEEDS
ROBUSTNESS_SEEDS = (3101, 3102, 3103)


def _precision_at_15(
    y: np.ndarray, probabilities: np.ndarray, ids: np.ndarray
) -> float:
    score, _ = precision_at_k(y, probabilities, ids)
    return float(score)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.Series,
    *,
    predictors: tuple[str, ...],
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    if (
        tuple(predictors) != ALLOWED_PREDICTORS
        or tuple(X.columns) != ALLOWED_PREDICTORS
    ):
        raise ValueError("El experimento requiere exactamente las features permitidas")

    candidates = {
        "logistic_l2": [],
        "stacking_lgbm_xgb": [],
        "stacking_lgbm_xgb_catboost": [],
    }
    predictions = {name: np.zeros(len(X), dtype=float) for name in candidates}
    counts = np.zeros(len(X), dtype=int)
    id_values = ids.to_numpy(dtype=str)
    y_values = y.to_numpy(dtype=int)

    for seed in seeds:
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(outer_cv.split(X, y), start=1):
            fold_seed = seed * 10 + fold
            models = {
                "logistic_l2": make_pipeline(C=0.1),
                "stacking_lgbm_xgb": make_stacking_pipeline(random_state=fold_seed),
                "stacking_lgbm_xgb_catboost": make_stacking_pipeline(
                    random_state=fold_seed, include_catboost=True
                ),
            }
            for name, model in models.items():
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                probabilities = model.predict_proba(X.iloc[valid_idx])[:, 1]
                predictions[name][valid_idx] += probabilities
                candidates[name].append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "precision_at_15": _precision_at_15(
                            y_values[valid_idx],
                            probabilities,
                            id_values[valid_idx],
                        ),
                    }
                )
            counts[valid_idx] += 1

    results = []
    for name, fold_results in candidates.items():
        pooled_probabilities = predictions[name] / counts
        fold_scores = [row["precision_at_15"] for row in fold_results]
        results.append(
            {
                "model": name,
                "precision_at_15": _precision_at_15(
                    y_values, pooled_probabilities, id_values
                ),
                "fold_precision_mean": float(np.mean(fold_scores)),
                "fold_precision_sd": float(np.std(fold_scores, ddof=1)),
                "folds": fold_results,
            }
        )

    return {
        "metric": "precision_at_15",
        "capacity_fraction": 0.15,
        "features": list(predictors),
        "seeds": list(seeds),
        "outer_folds": 5,
        "stacking_inner_folds": 3,
        "stacking_meta_logistic_C": STACKING_META_C,
        "versions": {
            package: version(package)
            for package in ("scikit-learn", "lightgbm", "xgboost", "catboost")
        },
        "candidates": results,
    }


def main() -> None:
    from churnshield.data import load_dev

    dev, _ = load_dev()
    predictors = tuple(ALLOWED_PREDICTORS)
    X = dev[list(predictors)]
    y = dev["Churn Value"].astype(int)
    ids = dev["CustomerID"].astype(str)
    result = {
        "configured_seeds": compare_models(
            X, y, ids, predictors=predictors, seeds=SEEDS
        ),
        "robustness_seeds": compare_models(
            X, y, ids, predictors=predictors, seeds=ROBUSTNESS_SEEDS
        ),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "stacking_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )

    def summaries(comparison: dict) -> list[dict]:
        return [
            {
                key: candidate[key]
                for key in (
                    "model",
                    "precision_at_15",
                    "fold_precision_mean",
                    "fold_precision_sd",
                )
            }
            for candidate in comparison["candidates"]
        ]

    print(
        json.dumps(
            {
                "configured_seeds": summaries(result["configured_seeds"]),
                "robustness_seeds": summaries(result["robustness_seeds"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
