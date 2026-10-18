import csv

import pytest

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
    assert {"accepted", "rejected", "error", "crm_error"}.issubset(
        batch["m3"]["s10_statuses"]
    )


def test_evaluate_batch_rejects_empty_input_and_unexpected_runs(tmp_path):
    with pytest.raises(ValueError, match="No hay ejecuciones"):
        evaluate_batch(tmp_path)

    run_scenario("S02", seed=101, output_dir=tmp_path)
    with pytest.raises(ValueError, match="Ejecuciones inesperadas"):
        evaluate_batch(tmp_path, expected_runs={("S03", 101)})


def test_m2_rejects_duplicate_records(tmp_path):
    run_scenario("S02", seed=101, output_dir=tmp_path)
    run_dir = tmp_path / "S02" / "seed_101"
    decisions_path = run_dir / "decisions.csv"
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        first_record = next(reader)
    with decisions_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerow(first_record)

    with pytest.raises(ValueError, match="duplicada"):
        evaluate_m2(run_dir)


def test_m3_requires_actual_failure_evidence_for_s10(tmp_path):
    run_scenario("S10", seed=101, output_dir=tmp_path)
    outcomes_path = tmp_path / "S10" / "seed_101" / "outcomes.csv"
    with outcomes_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        outcomes = list(reader)
    for row in outcomes:
        if row["final_status"] in {"rejected", "error", "crm_error"}:
            row["final_status"] = "accepted"
    with outcomes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outcomes)

    result = evaluate_m3(tmp_path / "S10" / "seed_101")

    assert result["checks"]["failure_statuses_registered"] is False
    assert result["passes"] is False
