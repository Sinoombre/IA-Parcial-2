import numpy as np
import pytest

from churnshield.simulation import cohort
from churnshield.simulation.cohort import simulate_cohort


def test_simulation_is_reproducible_and_hides_truth(small_df, simulation_inputs):
    p, q = simulation_inputs
    public1, truth1 = simulate_cohort(
        small_df, p, q, repetition_seed=101, cycles=3, r_true=1.0
    )
    public2, truth2 = simulate_cohort(
        small_df, p, q, repetition_seed=101, cycles=3, r_true=1.0
    )

    assert public1 == public2
    assert truth1 == truth2
    assert public1
    assert all(
        "q_i" not in row and "would_churn" not in row and "saved" not in row
        for row in public1
    )
    assert all(
        {"q_i", "would_churn", "saved", "r_true"}.issubset(row) for row in truth1
    )


def test_cycle_one_uses_empirical_churn_and_absorbs_cancelled_clients(
    small_df, simulation_inputs
):
    p, q = simulation_inputs
    public, truth = simulate_cohort(
        small_df, p, q, repetition_seed=101, cycles=2, r_true=0.0
    )
    cycle_one = {row["customer_id"]: row for row in truth if row["cycle"] == 1}
    cycle_two_ids = {row["customer_id"] for row in public if row["cycle"] == 2}

    assert cycle_one["A"]["would_churn"] == 1
    assert cycle_one["D"]["would_churn"] == 1
    assert "A" not in cycle_two_ids
    assert "D" not in cycle_two_ids


def test_run_simulation_writes_evaluator_truth_separately(
    tmp_path, small_df, monkeypatch
):
    ids = small_df["CustomerID"].astype(str).tolist()
    inputs = (
        small_df,
        dict(zip(ids, [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])),
        dict(zip(ids, [0.2] * len(ids))),
        {
            "simulation_truth_source": "development_only_oof_oracle",
            "simulation_propensity_mean": 0.2,
        },
    )
    monkeypatch.setattr(cohort, "development_inputs", lambda: inputs)

    result = cohort.run_simulation(output_dir=tmp_path / "public")

    public_dir = tmp_path / "public"
    evaluator_file = (
        tmp_path.parent
        / f"{tmp_path.name}-private-evaluator-truth"
        / "cohort_simulation"
        / "evaluator_truth.csv"
    )
    assert (public_dir / "cohort_public.csv").exists()
    assert not (public_dir / "evaluator_truth.csv").exists()
    assert evaluator_file.exists()
    assert result["truth_file"] == str(evaluator_file)


def test_public_artifact_output_routes_evaluator_truth_to_ignored_tree(tmp_path):
    from churnshield.simulation.paths import evaluator_output_path

    output_dir = tmp_path / "artifacts" / "public" / "scenarios"

    assert (
        evaluator_output_path(output_dir, None)
        == tmp_path.parent / f"{tmp_path.name}-private-evaluator-truth"
    )


def test_evaluator_output_rejects_public_or_unrelated_destinations(
    tmp_path, monkeypatch
):
    from churnshield.simulation import paths

    output_dir = tmp_path / "artifacts" / "public" / "scenarios"
    private_root = tmp_path / "artifacts" / "private" / "evaluator_truth"
    monkeypatch.setattr(paths, "PRIVATE_EVALUATOR_ROOT", private_root)

    for destination in (
        tmp_path / "artifacts" / "public" / "truth",
        tmp_path / "artifacts" / "private" / "other",
        tmp_path / "outside" / "evaluator_truth",
    ):
        with pytest.raises(ValueError, match="artifacts/private/evaluator_truth"):
            paths.evaluator_output_path(output_dir, destination)


def test_evaluator_output_allows_only_private_truth_tree(tmp_path, monkeypatch):
    from churnshield.simulation import paths

    output_dir = tmp_path / "artifacts" / "public" / "scenarios"
    private_root = tmp_path / "artifacts" / "private" / "evaluator_truth"
    monkeypatch.setattr(paths, "PRIVATE_EVALUATOR_ROOT", private_root)
    destination = private_root / "scenarios"

    assert paths.evaluator_output_path(output_dir, destination) == destination


def test_development_inputs_generates_oracle_truth_from_development_only(
    small_df, monkeypatch
):
    import pandas as pd

    from churnshield import data
    from churnshield.features import ALLOWED_PREDICTORS

    development = pd.DataFrame(
        [
            {
                **{feature: 1 for feature in ALLOWED_PREDICTORS},
                "CustomerID": customer_id,
                "CLTV": 100.0 + index,
                "Churn Value": index % 2,
                "Gender": "Female",
                "Senior Citizen": 0,
                "Country": "US",
                "State": "CA",
                "City": "City",
                "Zip Code": 90000 + index,
                "Lat Long": "0,0",
                "Latitude": 0.0,
                "Longitude": 0.0,
            }
            for index, customer_id in enumerate(small_df["CustomerID"])
        ]
    )
    calls = []
    monkeypatch.setattr(cohort, "load_dev", lambda: (development, {"dev_count": 7}))
    monkeypatch.setattr(cohort, "select_C", lambda *args: (0.1, None, None))
    monkeypatch.setattr(
        cohort, "oof_predictions", lambda X, y, ids, C: np.full(len(X), 0.4)
    )

    def oracle_for_development(frame, *, seed, target_mean):
        calls.append((frame, seed, target_mean))
        return np.full(len(frame), target_mean)

    monkeypatch.setattr(cohort, "oracle_oof_predictions", oracle_for_development)
    monkeypatch.setattr(
        data,
        "_read_data_rows",
        lambda indices, *, split: pytest.fail(
            f"Unexpected dataset split access: {split}"
        ),
    )
    monkeypatch.setattr(
        data,
        "load_reserved",
        lambda **kwargs: pytest.fail("simulate must not access reserved data"),
    )

    result = cohort.development_inputs()

    assert len(calls) == 1
    assert calls[0][0] is development
    assert len(result[0]) == 7
    assert set(result[2]) == set(development["CustomerID"])
    assert result[3]["reserved_access"] is False
    assert result[3]["simulation_truth_source"] == "development_only_oof_oracle"
