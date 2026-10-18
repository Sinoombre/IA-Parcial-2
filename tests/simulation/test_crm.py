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


@pytest.mark.parametrize(
    "rates",
    [
        {"timeout_rate": -0.1},
        {"rejected_rate": 1.1},
        {"timeout_rate": 0.7, "error_rate": 0.4},
        {"error_rate": float("nan")},
        {"timeout_rate": "invalid"},
    ],
)
def test_crm_rejects_invalid_outcome_rates(rates):
    with pytest.raises(ValueError, match="tasas"):
        SimulatedCRM(**rates)


def test_crm_rejects_campaign_reuse_for_another_customer():
    crm = SimulatedCRM()
    crm.send("campaign-1", "A", np.random.default_rng(1))

    with pytest.raises(ValueError, match="otro cliente"):
        crm.send("campaign-1", "B", np.random.default_rng(2))


def test_crm_rejects_blank_identifiers():
    crm = SimulatedCRM()

    with pytest.raises(ValueError, match="campaign_id"):
        crm.send("", "A", np.random.default_rng(1))
    with pytest.raises(ValueError, match="customer_id"):
        crm.send("campaign-1", " ", np.random.default_rng(1))
    with pytest.raises(ValueError, match="customer_id"):
        crm.send("campaign-2", None, np.random.default_rng(1))


@pytest.mark.parametrize("retry_count", [True, 1.5, None, -1])
def test_crm_rejects_invalid_retry_counts(retry_count):
    with pytest.raises(ValueError):
        SimulatedCRM(timeout_retries=retry_count)
