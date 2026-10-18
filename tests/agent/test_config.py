from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "agent.yaml"


def test_config_has_initial_agent_contract():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["selection"]["capacity_fraction"] == 0.15
    assert config["selection"]["capacity_rule"].startswith("floor")
    assert config["agent_run"]["scenario"] == "S03"
    assert config["agent_run"]["models_trained"] is False
    assert config["simulation"]["status"] == "implemented"
    assert config["learning"]["status"] == "implemented"
    assert config["crm"]["status"] == "implemented"
    assert config["agent_run"]["simulator_implemented"] is True
    assert config["agent_run"]["learning_implemented"] is True
    assert len(config["features"]["allowed_predictors"]) == 9
    assert set(config["features"]["excluded_by_design"]) == {
        "Phone Service",
        "Multiple Lines",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
    }
