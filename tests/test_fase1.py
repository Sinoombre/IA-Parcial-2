from pathlib import Path

import yaml

from src.churnshield.decision import select_candidates
from src.churnshield.s03 import run, synthetic_rows


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "agente.yaml"


def test_config_has_phase1_contract():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["selection"]["capacity_fraction"] == 0.15
    assert config["selection"]["capacity_rule"].startswith("floor")
    assert config["phase1"]["scenario"] == "S03"
    assert config["phase1"]["models_trained"] is False
    assert config["phase1"]["simulator_implemented"] is False


def test_s03_capacity_positive_utility_and_reproducibility():
    first = run()
    second = run()
    assert first == second
    assert first["k"] == 7
    assert first["n_selected"] <= 7
    assert first["coverage"] <= 0.15
    assert all(row["u_act"] > 0 for row in first["decisions"] if row["selected"])


def test_all_non_positive_utilities_select_nothing():
    rows = [
        {"customer_id": "A", "p_churn": 0.9, "u_act": 0.0},
        {"customer_id": "B", "p_churn": 0.8, "u_act": -0.1},
    ]
    result = select_candidates(rows)
    assert result["k"] == 0
    assert result["n_selected"] == 0
    assert result["coverage"] == 0.0


def test_tie_break_is_u_then_probability_then_customer_id():
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


def test_s03_contains_fifty_synthetic_rows():
    assert len(synthetic_rows()) == 50
