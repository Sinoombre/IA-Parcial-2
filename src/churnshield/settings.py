"""Repository configuration loaded from the canonical YAML files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_CONFIG_PATH = REPO_ROOT / "config" / "agent.yaml"
THRESHOLDS_CONFIG_PATH = REPO_ROOT / "config" / "thresholds.yaml"


@lru_cache(maxsize=1)
def agent_config() -> dict[str, Any]:
    with AGENT_CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config/agent.yaml debe contener un objeto YAML")
    return config


def threshold_config() -> dict[str, Any]:
    with THRESHOLDS_CONFIG_PATH.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config/thresholds.yaml debe contener un objeto YAML")
    return config
