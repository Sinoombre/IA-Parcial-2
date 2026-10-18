from churnshield.simulation.phase3 import simulate_cohort


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
    assert all({"q_i", "would_churn", "saved", "r_true"}.issubset(row) for row in truth1)


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
