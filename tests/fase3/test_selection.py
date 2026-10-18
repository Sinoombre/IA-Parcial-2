from churnshield.decision import select_candidates


def test_selection_never_exceeds_capacity():
    rows = [
        {"customer_id": str(i), "p_churn": 0.5, "u_act": 1.0}
        for i in range(50)
    ]
    result = select_candidates(rows)

    assert result["k"] == 7
    assert result["n_selected"] <= 7
