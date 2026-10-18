from churnshield.simulation.scenario_runner import run_scenario


def test_s10_runs_expanded_cohort(tmp_path):
    summary = run_scenario("S10", seed=101, output_dir=tmp_path)
    assert summary["scenario"] == "S10"
    assert summary["records"] == 100
