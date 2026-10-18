import pytest

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
    assert [row["customer_id"] for row in result["decisions"] if row["selected"]] == [
        "C",
        "A",
    ]


def test_duplicate_customer_ids_are_rejected_before_capacity_selection():
    rows = [
        {"customer_id": "A", "p_churn": 0.9, "u_act": 1.0},
        {"customer_id": "A", "p_churn": 0.8, "u_act": 0.9},
    ]

    with pytest.raises(ValueError, match="customer_id debe ser único"):
        select_candidates(rows, capacity_fraction=0.5)


@pytest.mark.parametrize(
    ("rows", "fraction", "message"),
    [
        ([{"customer_id": "A", "p_churn": 1.1, "u_act": 1.0}], 1.0, "p_churn"),
        (
            [{"customer_id": "A", "p_churn": 0.5, "u_act": float("nan")}],
            1.0,
            "u_act",
        ),
        (
            [{"customer_id": "A", "p_churn": 0.5, "u_act": 1.0}],
            float("nan"),
            "capacity_fraction",
        ),
        ([{"customer_id": "A", "p_churn": 0.5}], 1.0, "requiere"),
    ],
)
def test_candidate_selection_rejects_invalid_inputs(rows, fraction, message):
    with pytest.raises(ValueError, match=message):
        select_candidates(rows, capacity_fraction=fraction)


def test_unknown_ranking_fails_even_for_empty_candidates():
    with pytest.raises(ValueError, match="ranking"):
        select_candidates([], ranking="unknown")
