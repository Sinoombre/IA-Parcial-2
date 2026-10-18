from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "agente.yaml"


def test_config_has_phase1_contract():
    config = yaml.safe_load(CONFIG.read_text())
    assert config["selection"]["capacity_fraction"] == 0.15
    assert config["selection"]["capacity_rule"].startswith("floor")
    assert config["phase1"]["scenario"] == "S03"
    assert config["phase1"]["models_trained"] is False
    assert config["phase1"]["simulator_implemented"] is False
