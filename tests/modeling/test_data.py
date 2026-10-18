import pytest

from churnshield import data
from churnshield.data import load_dev, load_reserved


def test_reserved_is_blocked_without_flag():
    with pytest.raises(PermissionError):
        load_reserved()


def test_dataset_dev_contract(monkeypatch):
    original_sha256 = data.dataset_sha256
    hash_calls = 0

    def count_hash_calls(path=None):
        nonlocal hash_calls
        hash_calls += 1
        return original_sha256(path)

    monkeypatch.setattr(data, "dataset_sha256", count_hash_calls)
    df, meta = load_dev()
    assert len(df) == 5634
    assert meta["reserved_count"] == 1409
    assert meta["total_charges_blank_count"] == 11
    assert hash_calls == 1


def test_dataset_hash_mismatch_raises_value_error(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "dataset_path", lambda: tmp_path / "dataset.xlsx")
    monkeypatch.setattr(data, "dataset_sha256", lambda _: "unexpected")

    with pytest.raises(ValueError, match="SHA-256 del dataset"):
        data._read_and_validate()


def test_load_dev_and_full_reads_and_hashes_dataset_once(monkeypatch):
    original_read_excel = data.pd.read_excel
    original_sha256 = data.dataset_sha256
    read_calls = 0
    hash_calls = 0

    def count_reads(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original_read_excel(*args, **kwargs)

    def count_hashes(path=None):
        nonlocal hash_calls
        hash_calls += 1
        return original_sha256(path)

    monkeypatch.setattr(data.pd, "read_excel", count_reads)
    monkeypatch.setattr(data, "dataset_sha256", count_hashes)
    development, full, metadata = data.load_dev_and_full()

    assert len(development) == 5634
    assert len(full) == 7043
    assert metadata["dataset_sha256"]
    assert read_calls == hash_calls == 1
