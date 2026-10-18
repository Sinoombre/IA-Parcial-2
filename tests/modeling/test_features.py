import pytest

from churnshield.features import validate_observation


def test_forbidden_observation_fields_fail():
    with pytest.raises(ValueError):
        validate_observation(["Partner", "Churn Value"])
    with pytest.raises(ValueError):
        validate_observation(["Partner", "would_churn"])
