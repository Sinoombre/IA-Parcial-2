import csv
from collections import Counter

import numpy as np

from churnshield.simulation.crm import SimulatedCRM
from churnshield.simulation.scenario_runner import run_scenario, update_r_hat


def test_learning_formula_uses_only_accepted_treatments():
    value = update_r_hat(
        r0=0.25,
        m0=10,
        treated_probabilities=[0.8, 0.6],
        observed_churn=[0, 1],
    )
    assert 0.0 <= value <= 1.0
    assert value == (10 * 0.25 + 1.4 - 1) / (10 + 1.4)


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


def test_scenario_logs_segments_and_utc_timestamps(tmp_path):
    run_scenario("S02", seed=101, output_dir=tmp_path)
    with (tmp_path / "S02" / "seed_101" / "decisions.csv").open(newline="") as f:
        row = next(csv.DictReader(f))
    assert row["segment"] in {"high", "medium", "low"}
    assert row["timestamp_utc"].endswith("Z")
