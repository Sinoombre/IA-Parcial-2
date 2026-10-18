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
        {"oracle_population_mean": 0.2},
    )
    monkeypatch.setattr(cohort, "development_inputs", lambda: inputs)

    result = cohort.run_simulation(output_dir=tmp_path / "public")

    public_dir = tmp_path / "public"
    evaluator_file = (
        tmp_path / "evaluator_truth" / "cohort_simulation" / "evaluator_truth.csv"
    )
    assert (public_dir / "cohort_public.csv").exists()
    assert not (public_dir / "evaluator_truth.csv").exists()
    assert evaluator_file.exists()
    assert result["truth_file"] == str(evaluator_file)
