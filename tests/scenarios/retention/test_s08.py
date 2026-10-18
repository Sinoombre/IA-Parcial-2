from churnshield.scenarios.retention import SCENARIO_CONFIGS
from churnshield.simulation.scenario_runner import run_scenario


def test_s08_imputes_missing_cltv(tmp_path):
    assert SCENARIO_CONFIGS["S08"].missing_first_cltv is True
    assert run_scenario("S08", seed=101, output_dir=tmp_path)["scenario"] == "S08"
