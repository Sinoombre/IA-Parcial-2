"""Oráculo oculto del simulador.

El oráculo solo sirve para generar q_i del evaluador. Nunca se entrega a
perception.observe ni participa en la selección de campañas.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ORACLE_FEATURES = [
    "Gender", "Senior Citizen", "CLTV", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
]
NUMERIC = ["CLTV", "Zip Code", "Latitude", "Longitude"]


def _oracle_pipeline(seed: int) -> Pipeline:
    
    categorical = [c for c in ORACLE_FEATURES if c not in NUMERIC]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, random_state=seed,
        n_jobs=1, class_weight="balanced_subsample",
    )

    return Pipeline([("preprocess", prep), ("model", model)])


def _calibrate_mean(prob: np.ndarray, target_mean: float) -> np.ndarray:
    """Ajusta solo la media de q_i mediante un corrimiento logit.

    Esto no se presenta como calibración probabilística, conserva el orden y
    fuerza la media de la población inicial a la referencia configurada.
    """

    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    target = float(np.clip(target_mean, 1e-6, 1 - 1e-6))
    lo, hi = -20.0, 20.0
    logits = np.log(p / (1.0 - p))

    for _ in range(80):
        shift = (lo + hi) / 2.0
        mean = float(np.mean(1.0 / (1.0 + np.exp(-(logits + shift)))))
        if mean < target:
            lo = shift
        else:
            hi = shift

    out = 1.0 / (1.0 + np.exp(-(logits + (lo + hi) / 2.0)))
    return np.clip(out, 1e-6, 1 - 1e-6)


def oracle_oof_predictions(
    df: pd.DataFrame,
    *,
    seed: int = 1702,
    target_mean: float = 0.2654,
    n_splits: int = 5,
) -> np.ndarray:
    
    """Devuelve q_i OOF para toda la población y calibra su media.

    Las etiquetas se usan exclusivamente para crear la verdad oculta del
    simulador. La función no es llamada por el agente ni por la política.
    """
    missing = [c for c in ORACLE_FEATURES + ["Churn Value"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas del oráculo: {missing}")
    X, y = df[ORACLE_FEATURES].copy(), df["Churn Value"].astype(int)
    pred = np.zeros(len(df), dtype=float)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, valid in cv.split(X, y):

        pipe = _oracle_pipeline(seed)
        pipe.fit(X.iloc[train], y.iloc[train])
        pred[valid] = pipe.predict_proba(X.iloc[valid])[:, 1]

    return _calibrate_mean(pred, target_mean)


def subset_by_customer_id(df: pd.DataFrame, customer_ids: Iterable[str], q_all: np.ndarray) -> dict[str, float]:
    """Devuelve q_i para un subconjunto de clientes."""
    if len(df) != len(q_all):
        raise ValueError("q_all debe tener una predicción por fila del dataframe")
    return dict(zip(df["CustomerID"].astype(str), np.asarray(q_all, dtype=float)))
