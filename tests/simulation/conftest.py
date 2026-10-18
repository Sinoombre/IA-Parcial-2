import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def small_df():
    return pd.DataFrame({
        "CustomerID": ["A", "B", "C", "D", "E", "F", "G"],
        "CLTV": [100, 200, 300, 400, 500, 600, 700],
        "Churn Value": [1, 0, 0, 1, 0, 0, 0],
    })


@pytest.fixture
def simulation_inputs(small_df):
    ids = small_df["CustomerID"].tolist()
    probabilities = dict(zip(ids, np.linspace(0.8, 0.2, len(ids))))
    propensities = dict(zip(ids, np.linspace(0.2, 0.8, len(ids))))
    return probabilities, propensities
