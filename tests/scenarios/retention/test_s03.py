from churnshield.simulation.scenario_runner import run_scenario


def test_s03_runs_standard_cohort(tmp_path):
    summary = run_scenario("S03", seed=101, output_dir=tmp_path)
    assert summary["scenario"] == "S03"
    assert summary["records"] == 50
