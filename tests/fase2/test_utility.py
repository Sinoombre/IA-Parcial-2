import pytest

from churnshield.utility import action_utility


def test_utility_formula():
    assert action_utility(0.8, 1.2, r=0.25, cost=0.05) == pytest.approx(0.19)
