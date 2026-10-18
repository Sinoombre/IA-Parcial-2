import numpy as np
import pytest

from churnshield.simulation.crm import SimulatedCRM


def test_timeout_has_one_retry_and_is_idempotent():
    crm = SimulatedCRM(timeout_rate=1.0, timeout_retries=1)
    first = crm.send("campaign-1", "A", np.random.default_rng(1))
    second = crm.send("campaign-1", "A", np.random.default_rng(999))

    assert first.outcome == "timeout"
    assert first.attempts == 2
    assert second == first
    assert len(crm.results) == 1


def test_reusing_campaign_id_for_a_different_customer_is_rejected():
    crm = SimulatedCRM()
    crm.send("campaign-1", "A", np.random.default_rng(1))

    with pytest.raises(ValueError, match="otro customer_id"):
        crm.send("campaign-1", "B", np.random.default_rng(2))


@pytest.mark.parametrize(
    "timeout_retries",
    [-1, 1.5, float("nan"), float("inf"), "not-an-integer"],
)
def test_timeout_retries_must_be_a_nonnegative_integer(timeout_retries):
    with pytest.raises(ValueError, match="timeout_retries"):
        SimulatedCRM(timeout_retries=timeout_retries)


@pytest.mark.parametrize(
    ("rates", "message"),
    [
        ({"timeout_rate": -0.1}, "tasas del CRM"),
        ({"rejected_rate": 1.1}, "tasas del CRM"),
        (
            {"timeout_rate": 0.6, "rejected_rate": 0.5},
            "tasas del CRM",
        ),
        ({"error_rate": float("nan")}, "tasas del CRM"),
    ],
)
def test_crm_rejects_invalid_outcome_rates(rates, message):
    with pytest.raises(ValueError, match=message):
        SimulatedCRM(**rates)
