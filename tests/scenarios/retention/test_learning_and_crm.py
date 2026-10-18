import csv
from collections import Counter

import numpy as np
import pytest

from churnshield.simulation.crm import SimulatedCRM
from churnshield.simulation.scenario_runner import (
    _timestamp,
    run_many,
    run_scenario,
    update_r_hat,
)


def test_learning_formula_uses_only_accepted_treatments():
    value = update_r_hat(
        r0=0.25,
        m0=10,
        treated_probabilities=[0.8, 0.6],
        observed_churn=[0, 1],
    )
    assert 0.0 <= value <= 1.0
    assert value == (10 * 0.25 + 1.4 - 1) / (10 + 1.4)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"r0": 1.1}, "r0"),
        ({"m0": 0}, "m0"),
        ({"treated_probabilities": [1.1]}, "probabilidades"),
        ({"observed_churn": [0.5]}, "churn observado"),
    ],
)
def test_learning_formula_rejects_invalid_inputs(overrides, message):
    arguments = {
        "r0": 0.25,
        "m0": 10,
        "treated_probabilities": [0.8],
        "observed_churn": [0],
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        update_r_hat(**arguments)


def test_synthetic_timestamps_use_a_fixed_utc_timeline():
    assert _timestamp(1, 0) == "2026-01-01T00:00:00Z"
    assert _timestamp(2, 3) == "2026-01-02T00:00:03Z"


def test_crm_second_timeout_is_recorded_as_crm_error():
    crm = SimulatedCRM(timeout_rate=1.0, timeout_retries=1)
    result = crm.send("c1", "A", np.random.default_rng(1))
    assert result.first_outcome == "timeout"
    assert result.second_outcome == "timeout"
    assert result.final_status == "crm_error"
    assert result.retry_used is True


def test_s10_has_varied_crm_results_and_retry_logs(tmp_path):
    summary = run_scenario("S10", seed=101, output_dir=tmp_path)
    with (tmp_path / "S10" / "seed_101" / "outcomes.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    statuses = Counter(row["final_status"] for row in rows)
    assert statuses["not_sent"] > 0
    assert any(
        status in statuses
        for status in ("accepted", "rejected", "error", "crm_error", "timeout")
    )
    assert (tmp_path / "S10" / "seed_101" / "perceptions.jsonl").exists()
    assert summary["scenario"] == "S10"


def test_s10_five_seeds_cover_all_expected_failure_statuses(tmp_path):
    run_many(("S10",), output_dir=tmp_path)
    statuses = Counter()
    retries = 0
    for seed in (101, 102, 103, 104, 105):
        with (tmp_path / "S10" / f"seed_{seed}" / "outcomes.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        statuses.update(row["final_status"] for row in rows)
        retries += sum(int(row["retry_used"]) for row in rows)
    assert {"accepted", "rejected", "error", "crm_error"}.issubset(statuses)
    assert retries > 0


def test_scenario_logs_segments_and_utc_timestamps(tmp_path):
    run_scenario("S02", seed=101, output_dir=tmp_path)
    with (tmp_path / "S02" / "seed_101" / "decisions.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["segment"] in {"high", "medium", "low"}
    assert row["timestamp_utc"].endswith("Z")


def test_run_many_reuses_generator_seeds_for_every_scenario(tmp_path):
    summaries = run_many(
        iter(("S04", "S06")),
        seeds=(seed for seed in (101, 102)),
        output_dir=tmp_path / "public",
        evaluator_output_dir=tmp_path / "truth",
    )
    assert len(summaries) == 4
    assert (tmp_path / "public" / "scenario_summary.csv").exists()
    assert (tmp_path / "truth" / "S04" / "seed_102" / "evaluator_truth.csv").exists()


def test_public_and_evaluator_output_paths_must_not_overlap(tmp_path):
    with pytest.raises(ValueError, match="carpetas separadas"):
        run_scenario(
            "S02",
            output_dir=tmp_path,
            evaluator_output_dir=tmp_path / "truth",
        )
