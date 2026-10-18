"""Frontera de observación del agente."""

from __future__ import annotations
import pandas as pd
from .features import ALLOWED_PREDICTORS, validate_observation


def observe(snapshot: pd.DataFrame) -> pd.DataFrame:

    validate_observation(snapshot.columns)
    missing = [c for c in ALLOWED_PREDICTORS if c not in snapshot.columns]
    if missing:
        raise ValueError(f"Faltan predictores permitidos: {missing}")
    return snapshot[list(ALLOWED_PREDICTORS)].copy()
