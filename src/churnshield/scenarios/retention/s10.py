"""S10: cohorte ampliada con fallos y reintentos del CRM."""

from .scenario import Scenario

SCENARIO = Scenario(
    scenario_id="S10",
    customer_count=100,
    crm_timeout_rate=0.20,
    crm_rejected_rate=0.10,
    crm_error_rate=0.10,
)
