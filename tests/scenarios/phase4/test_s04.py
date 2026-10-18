from churnshield.simulation.phase4 import run_scenario


def test_s04_runs_small_cohort(tmp_path):
    summary = run_scenario("S04", seed=101, output_dir=tmp_path)
    assert summary["scenario"] == "S04"
    assert summary["records"] == 6
