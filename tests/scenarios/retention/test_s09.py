from churnshield.simulation.scenario_runner import run_scenario


def test_s09_marks_unknown_category(tmp_path):
    summary = run_scenario("S09", seed=101, output_dir=tmp_path)
    assert summary["scenario"] == "S09"
    assert summary["unknown_category"] is True
    assert summary["unknown_category_handled"] is True
    assert 0.0 <= summary["probe_probability"] <= 1.0
