"""Construcción reproducible de los datos sintéticos de retención."""

import numpy as np
import pandas as pd

from .scenario import Scenario


def build_fixture(
    scenario: Scenario, seed: int
) -> tuple[pd.DataFrame, dict[str, float], dict[str, float], dict]:
    ids = [f"F4-{i:03d}" for i in range(scenario.customer_count)]
    rng = np.random.default_rng(seed)
    probabilities = np.linspace(0.9, 0.1, scenario.customer_count)
    cltv = np.linspace(100.0, 1000.0, scenario.customer_count)
    historical = (rng.random(scenario.customer_count) < scenario.churn_rate).astype(int)

    if scenario.probability_pattern == "equal":
        probabilities[:] = 0.5
    elif scenario.probability_pattern == "low":
        probabilities[:] = 0.2

    if scenario.cltv_pattern == "equal":
        cltv[:] = 500.0
    elif scenario.cltv_pattern == "low":
        cltv[:] = 100.0

    if scenario.missing_first_cltv:
        cltv[0] = np.nan

    df = pd.DataFrame(
        {"CustomerID": ids, "CLTV": cltv, "Churn Value": historical}
    )
    if scenario.missing_first_cltv:
        df["CLTV"] = df["CLTV"].fillna(df["CLTV"].median())

    propensities = np.full(scenario.customer_count, scenario.churn_rate)
    return (
        df,
        dict(zip(ids, probabilities)),
        dict(zip(ids, propensities)),
        {
            "scenario": scenario.scenario_id,
            "synthetic": True,
            "unknown_category": scenario.unknown_category,
        },
    )
