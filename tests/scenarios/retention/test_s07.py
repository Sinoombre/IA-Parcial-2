from churnshield.scenarios.retention import SCENARIO_CONFIGS
from churnshield.simulation.scenario_runner import run_scenario


def test_s07_uses_low_churn_rate(tmp_path):
    assert SCENARIO_CONFIGS["S07"].churn_rate == 0.05
    assert run_scenario("S07", seed=101, output_dir=tmp_path)["scenario"] == "S07"
