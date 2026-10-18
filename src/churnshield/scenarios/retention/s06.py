"""S06: utilidad de acción reducida por costo elevado."""

from .scenario import Scenario

SCENARIO = Scenario(
    scenario_id="S06",
    cost=1.0,
    probability_pattern="low",
    cltv_pattern="low",
)
