from churnshield.scenarios.phase4 import SCENARIO_CONFIGS
from churnshield.simulation.phase4 import run_scenario


def test_s06_uses_high_action_cost(tmp_path):
    assert SCENARIO_CONFIGS["S06"].cost == 1.0
    assert run_scenario("S06", seed=101, output_dir=tmp_path)["scenario"] == "S06"
