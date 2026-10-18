import pytest

from churnshield.data import load_dev, load_reserved


def test_reserved_is_blocked_without_flag():
    with pytest.raises(PermissionError):
        load_reserved()


def test_dataset_dev_contract():
    df, meta = load_dev()
    assert len(df) == 5634
    assert meta["reserved_count"] == 1409
    assert meta["total_charges_blank_count"] == 11
