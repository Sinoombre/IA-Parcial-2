from churnshield.decision import select_candidates


def test_all_non_positive_utilities_select_nothing():
    rows = [
        {"customer_id": "A", "p_churn": 0.9, "u_act": 0.0},
        {"customer_id": "B", "p_churn": 0.8, "u_act": -0.1},
    ]
    result = select_candidates(rows)
    assert result["k"] == 0
    assert result["n_selected"] == 0
    assert result["coverage"] == 0.0


def test_utility_tie_break_is_u_then_probability_then_customer_id():
    rows = [
        {"customer_id": "B", "p_churn": 0.8, "u_act": 0.4},
        {"customer_id": "A", "p_churn": 0.8, "u_act": 0.4},
        {"customer_id": "C", "p_churn": 0.9, "u_act": 0.4},
        {"customer_id": "D", "p_churn": 0.9, "u_act": 0.3},
    ]
    result = select_candidates(rows, capacity_fraction=0.5)
    ranked = [row["customer_id"] for row in result["decisions"]]
    assert ranked == ["C", "A", "B", "D"]
    assert [row["customer_id"] for row in result["decisions"] if row["selected"]] == ["C", "A"]
