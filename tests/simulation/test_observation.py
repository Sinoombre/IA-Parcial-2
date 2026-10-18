import pytest

from churnshield.features import validate_observation


def test_agent_observation_rejects_hidden_truth():
    with pytest.raises(ValueError):
        validate_observation(["q_i"])
