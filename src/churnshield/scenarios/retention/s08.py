"""S08: CLTV incompleto con imputación por mediana."""

from .scenario import Scenario

SCENARIO = Scenario(scenario_id="S08", missing_first_cltv=True)
