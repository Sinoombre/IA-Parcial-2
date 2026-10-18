import pytest

from churnshield.simulation.phase4 import run_scenario


def test_s01_remains_blocked():
    with pytest.raises(PermissionError):
        run_scenario("S01", seed=101)
