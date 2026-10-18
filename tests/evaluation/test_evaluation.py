from churnshield.evaluation import evaluate_batch, evaluate_m2, evaluate_m3
from churnshield.simulation.scenario_runner import run_many, run_scenario


def test_m2_reconstructs_a_complete_run(tmp_path):

    run_scenario("S02", seed=101, output_dir=tmp_path)
    result = evaluate_m2(tmp_path / "S02" / "seed_101")
    assert result["passes"] is True
    assert result["score"] == 1.0


def test_m3_covers_failure_scenarios_and_batch(tmp_path):

    run_many(("S04", "S06", "S08", "S09", "S10"), output_dir=tmp_path)
    for scenario in ("S04", "S06", "S08", "S09", "S10"):
        result = evaluate_m3(tmp_path / scenario / "seed_101")
        assert result["passes"] is True, result
        
    batch = evaluate_batch(tmp_path)
    assert batch["m2"]["passes"] is True
    assert batch["m3"]["passes"] is True
    assert {"accepted", "rejected", "error", "crm_error"}.issubset(batch["m3"]["s10_statuses"])
