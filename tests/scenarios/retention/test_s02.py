import pytest

from churnshield import data
from churnshield.simulation.scenario_runner import run_scenario


def test_s02_runs_six_cycles(tmp_path):
    summary = run_scenario("S02", seed=101, output_dir=tmp_path)
    assert summary["scenario"] == "S02"
    assert summary["cycles"] == 6
    assert (tmp_path / "S02" / "seed_101" / "evaluator_truth.csv").exists() is False
    assert (
        tmp_path.parent / "evaluator_truth" / "S02" / "seed_101" / "evaluator_truth.csv"
    ).exists()


def test_scenarios_are_synthetic_and_do_not_load_customer_population(
    monkeypatch, tmp_path
):
    def reject_customer_data(*args, **kwargs):
        pytest.fail("scenario runner must not load the customer workbook")

    monkeypatch.setattr(data, "_read_data_rows", reject_customer_data)
    monkeypatch.setattr(data, "load_reserved", reject_customer_data)

    result = run_scenario(
        "S02",
        seed=101,
        output_dir=tmp_path / "public",
        evaluator_output_dir=tmp_path / "truth",
    )

    assert result["synthetic"] is True
