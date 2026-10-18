import pytest
import yaml

from churnshield.features import ALLOWED_PREDICTORS, validate_observation
from churnshield.settings import AGENT_CONFIG_PATH


def test_forbidden_observation_fields_fail():
    with pytest.raises(ValueError):
        validate_observation(["Partner", "Churn Value"])
    with pytest.raises(ValueError):
        validate_observation(["Partner", "would_churn"])


def test_runtime_features_match_the_single_yaml_contract():
    config = yaml.safe_load(AGENT_CONFIG_PATH.read_text(encoding="utf-8"))

    assert list(ALLOWED_PREDICTORS) == config["features"]["allowed_predictors"]
    assert len(ALLOWED_PREDICTORS) == 9
