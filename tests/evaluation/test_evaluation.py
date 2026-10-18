import csv
import json

from churnshield.evaluation import evaluate_batch, evaluate_m2, evaluate_m3
from churnshield.simulation.scenario_runner import run_many, run_scenario


def test_evaluation_does_not_pass_an_empty_batch(tmp_path):
    result = evaluate_batch(tmp_path)

    assert result["runs"] == 0
    assert result["m2"]["score"] == 0.0
    assert result["m2"]["passes"] is False
    assert result["m3"]["score"] == 0.0
    assert result["m3"]["passes"] is False


def test_m2_reconstructs_a_complete_run(tmp_path):

    run_scenario("S02", seed=101, output_dir=tmp_path)
    result = evaluate_m2(tmp_path / "S02" / "seed_101")
    assert result["passes"] is True
    assert result["score"] == 1.0


def test_m2_rejects_mismatched_and_duplicate_perceptions(tmp_path):
    run_dir = tmp_path / "S02" / "seed_101"
    run_scenario("S02", seed=101, output_dir=tmp_path)
    perception_file = run_dir / "perceptions.jsonl"
    original = perception_file.read_text(encoding="utf-8")
    lines = original.splitlines()

    altered = json.loads(lines[0])
    altered["p_churn"] += 0.01
    perception_file.write_text(
        "\n".join([json.dumps(altered), *lines[1:]]) + "\n",
        encoding="utf-8",
    )
    mismatched = evaluate_m2(run_dir)
    assert mismatched["passes"] is False
    assert mismatched["reconstructible_decisions"] < mismatched["total_decisions"]

    perception_file.write_text(original + lines[0] + "\n", encoding="utf-8")
    duplicated = evaluate_m2(run_dir)
    assert duplicated["passes"] is False
    assert duplicated["reconstructible_decisions"] < duplicated["total_decisions"]


def test_m3_does_not_pass_s10_without_recorded_failure_statuses(tmp_path):
    run_scenario("S10", seed=101, output_dir=tmp_path)
    run_dir = tmp_path / "S10" / "seed_101"
    outcomes_path = run_dir / "outcomes.csv"
    with outcomes_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        outcomes = list(reader)

    assert any(int(row["retry_used"]) for row in outcomes)
    for row in outcomes:
        row["final_status"] = "accepted" if int(row["attempts"]) else "not_sent"
    with outcomes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outcomes)

    result = evaluate_m3(run_dir)

    assert result["checks"]["timeout_retry_registered"] is True
    assert result["checks"]["failure_statuses_registered"] is False
    assert result["passes"] is False


def test_m3_covers_failure_scenarios_and_batch(tmp_path):

    run_many(("S04", "S06", "S08", "S09", "S10"), output_dir=tmp_path)
    for scenario in ("S04", "S06", "S08", "S09", "S10"):
        result = evaluate_m3(tmp_path / scenario / "seed_101")
        assert result["passes"] is True, result

    batch = evaluate_batch(tmp_path)
    assert batch["m2"]["passes"] is True
    assert batch["m3"]["passes"] is True
    assert {"accepted", "rejected", "error", "crm_error"}.issubset(
        batch["m3"]["s10_statuses"]
    )
