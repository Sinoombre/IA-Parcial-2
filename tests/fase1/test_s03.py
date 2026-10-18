from churnshield.s03 import run, synthetic_rows


def test_s03_capacity_positive_utility_and_reproducibility():
    first = run()
    second = run()
    assert first == second
    assert first["k"] == 7
    assert first["n_selected"] <= 7
    assert first["coverage"] <= 0.15
    assert all(row["u_act"] > 0 for row in first["decisions"] if row["selected"])


def test_s03_contains_fifty_synthetic_rows():
    assert len(synthetic_rows()) == 50
