from churnshield.scenarios.phase4 import SCENARIO_CONFIGS
from churnshield.simulation.phase4 import run_scenario


def test_s08_imputes_missing_cltv(tmp_path):
    assert SCENARIO_CONFIGS["S08"].missing_first_cltv is True
    assert run_scenario("S08", seed=101, output_dir=tmp_path)["scenario"] == "S08"
