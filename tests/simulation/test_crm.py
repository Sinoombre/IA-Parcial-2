import numpy as np

from churnshield.simulation.crm import SimulatedCRM


def test_timeout_has_one_retry_and_is_idempotent():
    crm = SimulatedCRM(timeout_rate=1.0, timeout_retries=1)
    first = crm.send("campaign-1", "A", np.random.default_rng(1))
    second = crm.send("campaign-1", "A", np.random.default_rng(999))

    assert first.outcome == "timeout"
    assert first.attempts == 2
    assert second == first
    assert len(crm.results) == 1
