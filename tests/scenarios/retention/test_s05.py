from churnshield.scenarios.retention import SCENARIO_CONFIGS
from churnshield.simulation.scenario_runner import run_scenario


def test_s05_uses_tied_probabilities_and_values(tmp_path):
    config = SCENARIO_CONFIGS["S05"]
    assert config.probability_pattern == "equal"
    assert config.cltv_pattern == "equal"
    assert run_scenario("S05", seed=101, output_dir=tmp_path)["scenario"] == "S05"
