"""Configuración declarativa compartida por los escenarios de retención."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    customer_count: int = 50
    churn_rate: float = 0.2654
    cycles: int = 1
    cost: float = 0.05
    probability_pattern: str = "descending"
    cltv_pattern: str = "descending"
    missing_first_cltv: bool = False
    unknown_category: bool = False
    crm_timeout_rate: float = 0.0
    crm_rejected_rate: float = 0.0
    crm_error_rate: float = 0.0
