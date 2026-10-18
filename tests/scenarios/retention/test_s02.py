from churnshield.simulation.scenario_runner import run_scenario


def test_s02_runs_six_cycles(tmp_path):
    summary = run_scenario("S02", seed=101, output_dir=tmp_path)
    assert summary["scenario"] == "S02"
    assert summary["cycles"] == 6
    assert (tmp_path / "S02" / "seed_101" / "evaluator_truth.csv").exists()
