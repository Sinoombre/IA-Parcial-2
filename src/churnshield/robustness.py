"""Pruebas de robustez del pipeline para registrar evidencia experimental."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import ALLOWED_PREDICTORS
from .model import make_pipeline


def _toy_frame(contract_values: list[str]) -> pd.DataFrame:
    rows = []
    for index, contract in enumerate(contract_values):
        rows.append(
            {
                "Partner": "Yes" if index % 2 else "No",
                "Dependents": "No",
                "Phone Service": "Yes",
                "Multiple Lines": "No phone service",
                "Internet Service": "DSL",
                "Online Security": "No",
                "Online Backup": "No",
                "Device Protection": "No",
                "Tech Support": "No",
                "Streaming TV": "No",
                "Streaming Movies": "No",
                "Contract": contract,
                "Paperless Billing": "Yes",
                "Payment Method": "Electronic check",
                "Tenure Months": 12 + index,
                "Monthly Charges": 70.0 + index,
                "Total Charges": 840.0 + index,
            }
        )
    return pd.DataFrame(rows, columns=ALLOWED_PREDICTORS)


def run_unknown_category_probe() -> dict:
    """Entrena con categorías conocidas y predice una categoría nueva.

    El resultado positivo solo significa que el pipeline continúa y devuelve una
    probabilidad finita; no implica que la categoría nueva tenga buen desempeño.
    """
    train = _toy_frame(["Month-to-month", "One year", "Month-to-month", "One year"])
    y = np.array([0, 1, 0, 1])
    probe = _toy_frame(["Two year"])
    pipeline = make_pipeline(C=1.0)
    pipeline.fit(train, y)
    probability = float(pipeline.predict_proba(probe)[0, 1])
    return {
        "unknown_category_value": "Two year",
        "unknown_category_handled": bool(np.isfinite(probability)),
        "probe_probability": probability,
    }
